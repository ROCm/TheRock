#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/common/execution_memory.h"
#include "lib/sim/isa/common/numeric_conversions.h"
#include "lib/sim/isa/common/wave_execution_state.h"
#include "lib/sim/isa/gfx1201/binary_decoder.h"
#include "lib/sim/isa/gfx1201/interpreter.h"

namespace {

constexpr std::uint16_t kImplicitVccPairSgprIndex = 248;
constexpr std::uint16_t kM0RegisterIndex = 124;
constexpr std::uint32_t kQuietNaNF16Bits = 0x00007e00u;
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

float BitsToFloat(std::uint32_t value) {
  float result = 0.0f;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

std::uint32_t PackF16Pair(float low, float high) {
  using mirage::sim::isa::FloatToHalf;
  return static_cast<std::uint32_t>(FloatToHalf(low)) |
         (static_cast<std::uint32_t>(FloatToHalf(high)) << 16);
}

std::uint32_t PackBf16Pair(float low, float high) {
  using mirage::sim::isa::FloatToBFloat16;
  return static_cast<std::uint32_t>(FloatToBFloat16(low)) |
         (static_cast<std::uint32_t>(FloatToBFloat16(high)) << 16);
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

std::array<std::uint32_t, 2> MakeSmem(std::uint32_t op,
                                      std::uint32_t sdata,
                                      std::uint32_t sbase_start,
                                      bool imm,
                                      std::uint32_t offset_or_soffset,
                                      bool soffset_en = false) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(0x30u) << 26;
  word |= static_cast<std::uint64_t>(sbase_start >> 1);
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

std::array<std::uint32_t, 2> MakeSmemPrefetchPcRel(std::uint32_t op,
                                                   std::int32_t ioffset,
                                                   std::uint32_t soffset,
                                                   std::int32_t sdata) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(0x30u) << 26;
  word |= static_cast<std::uint64_t>(op & 0xffu) << 18;
  word |= static_cast<std::uint64_t>(static_cast<std::uint32_t>(sdata) & 0x1fu)
          << 6;
  word |= static_cast<std::uint64_t>(static_cast<std::uint32_t>(ioffset) &
                                     0x00ffffffu)
          << 32;
  word |= static_cast<std::uint64_t>(soffset & 0x7fu) << 57;
  return {static_cast<std::uint32_t>(word),
          static_cast<std::uint32_t>(word >> 32)};
}

std::array<std::uint32_t, 2> MakeSmemBasePrefetch(std::uint32_t op,
                                                  std::uint32_t sbase_start,
                                                  std::int32_t ioffset,
                                                  std::uint32_t soffset,
                                                  std::int32_t sdata) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(0x30u) << 26;
  word |= static_cast<std::uint64_t>(sbase_start >> 1);
  word |= static_cast<std::uint64_t>(static_cast<std::uint32_t>(sdata) & 0x1fu)
          << 6;
  word |= static_cast<std::uint64_t>(op & 0xffu) << 18;
  word |= static_cast<std::uint64_t>(static_cast<std::uint32_t>(ioffset) &
                                     0x00ffffffu)
          << 32;
  word |= static_cast<std::uint64_t>(soffset & 0x7fu) << 57;
  return {static_cast<std::uint32_t>(word),
          static_cast<std::uint32_t>(word >> 32)};
}

std::array<std::uint32_t, 2> MakeSmemBufferLoad(std::uint32_t op,
                                                std::uint32_t sdst,
                                                std::uint32_t sbase_start,
                                                std::int32_t ioffset,
                                                std::uint32_t soffset) {
  std::uint64_t word = 0;
  word |= static_cast<std::uint64_t>(0x30u) << 26;
  word |= static_cast<std::uint64_t>(sbase_start >> 1);
  word |= static_cast<std::uint64_t>(sdst) << 6;
  word |= static_cast<std::uint64_t>(op & 0xffu) << 18;
  word |= static_cast<std::uint64_t>(static_cast<std::uint32_t>(ioffset) &
                                     0x00ffffffu)
          << 32;
  word |= static_cast<std::uint64_t>(soffset & 0x7fu) << 57;
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
  word |= static_cast<std::uint64_t>(static_cast<std::uint32_t>(offset) &
                                     0x1fffu) << 0;
  word |= static_cast<std::uint64_t>(2u) << 14;
  word |= static_cast<std::uint64_t>(op & 0x7fu) << 18;
  word |= static_cast<std::uint64_t>(addr & 0xffu) << 32;
  word |= static_cast<std::uint64_t>(data & 0xffu) << 40;
  word |= static_cast<std::uint64_t>(saddr & 0x7fu) << 48;
  word |= static_cast<std::uint64_t>(vdst & 0xffu) << 56;
  return {static_cast<std::uint32_t>(word),
          static_cast<std::uint32_t>(word >> 32)};
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

bool ExpectThreeOperandInstruction(
    const mirage::sim::isa::DecodedInstruction& instruction,
    std::string_view expected_opcode,
    mirage::sim::isa::OperandKind operand0_kind,
    std::uint32_t operand0_value_or_index,
    mirage::sim::isa::OperandKind operand1_kind,
    std::uint32_t operand1_value_or_index,
    mirage::sim::isa::OperandKind operand2_kind,
    std::uint32_t operand2_value_or_index) {
  using namespace mirage::sim::isa;
  if (instruction.opcode != expected_opcode || instruction.operand_count != 3u ||
      instruction.operands[0].kind != operand0_kind ||
      instruction.operands[1].kind != operand1_kind ||
      instruction.operands[2].kind != operand2_kind) {
    return false;
  }
  const bool operand0_matches =
      operand0_kind == OperandKind::kImm32
          ? instruction.operands[0].imm32 == operand0_value_or_index
          : instruction.operands[0].index ==
                static_cast<std::uint16_t>(operand0_value_or_index);
  const bool operand1_matches =
      operand1_kind == OperandKind::kImm32
          ? instruction.operands[1].imm32 == operand1_value_or_index
          : instruction.operands[1].index ==
                static_cast<std::uint16_t>(operand1_value_or_index);
  const bool operand2_matches =
      operand2_kind == OperandKind::kImm32
          ? instruction.operands[2].imm32 == operand2_value_or_index
          : instruction.operands[2].index ==
                static_cast<std::uint16_t>(operand2_value_or_index);
  return operand0_matches && operand1_matches && operand2_matches;
}

bool ExpectFourOperandInstruction(
    const mirage::sim::isa::DecodedInstruction& instruction,
    std::string_view expected_opcode,
    mirage::sim::isa::OperandKind operand0_kind,
    std::uint32_t operand0_value_or_index,
    mirage::sim::isa::OperandKind operand1_kind,
    std::uint32_t operand1_value_or_index,
    mirage::sim::isa::OperandKind operand2_kind,
    std::uint32_t operand2_value_or_index,
    mirage::sim::isa::OperandKind operand3_kind,
    std::uint32_t operand3_value_or_index) {
  using namespace mirage::sim::isa;
  if (instruction.opcode != expected_opcode || instruction.operand_count != 4u ||
      instruction.operands[0].kind != operand0_kind ||
      instruction.operands[1].kind != operand1_kind ||
      instruction.operands[2].kind != operand2_kind ||
      instruction.operands[3].kind != operand3_kind) {
    return false;
  }
  const bool operand0_matches =
      operand0_kind == OperandKind::kImm32
          ? instruction.operands[0].imm32 == operand0_value_or_index
          : instruction.operands[0].index ==
                static_cast<std::uint16_t>(operand0_value_or_index);
  const bool operand1_matches =
      operand1_kind == OperandKind::kImm32
          ? instruction.operands[1].imm32 == operand1_value_or_index
          : instruction.operands[1].index ==
                static_cast<std::uint16_t>(operand1_value_or_index);
  const bool operand2_matches =
      operand2_kind == OperandKind::kImm32
          ? instruction.operands[2].imm32 == operand2_value_or_index
          : instruction.operands[2].index ==
                static_cast<std::uint16_t>(operand2_value_or_index);
  const bool operand3_matches =
      operand3_kind == OperandKind::kImm32
          ? instruction.operands[3].imm32 == operand3_value_or_index
          : instruction.operands[3].index ==
                static_cast<std::uint16_t>(operand3_value_or_index);
  return operand0_matches && operand1_matches && operand2_matches &&
         operand3_matches;
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

bool ExpectReadfirstlaneSeedState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.lane_count == 32u && state.exec_mask == (1ULL << 31) &&
         state.sgprs[60] == 0xfeedbeefu && state.halted &&
         !state.waiting_on_barrier && state.pc == 1u;
}

bool ExpectMovrelSeedState(const mirage::sim::isa::WaveExecutionState& state) {
  return state.lane_count == 32u && state.sgprs[kM0RegisterIndex] == 0x00030001u &&
         state.vgprs[10][0] == 0x10101010u &&
         state.vgprs[10][1] == 0x20202020u &&
         state.vgprs[10][2] == 0x30303030u &&
         state.vgprs[10][3] == 0x40404040u &&
         state.vgprs[11][0] == 0xcafef00du &&
         state.vgprs[11][1] == 0xcafef00du &&
         state.vgprs[11][2] == 0x11111111u &&
         state.vgprs[11][3] == 0xcafef00du &&
         state.vgprs[12][0] == 0xaaaa0001u &&
         state.vgprs[12][1] == 0xbbbb0002u &&
         state.vgprs[12][2] == 0x12121212u &&
         state.vgprs[12][3] == 0xdddd0004u &&
         state.vgprs[20][0] == 0x20200000u &&
         state.vgprs[20][1] == 0x20200001u &&
         state.vgprs[20][2] == 0x20200002u &&
         state.vgprs[20][3] == 0x20200003u &&
         state.vgprs[21][0] == 0xaaaa0001u &&
         state.vgprs[21][1] == 0xbbbb0002u &&
         state.vgprs[21][2] == 0x41414141u &&
         state.vgprs[21][3] == 0xdddd0004u &&
         state.vgprs[30][0] == 0x30300000u &&
         state.vgprs[30][1] == 0x30300001u &&
         state.vgprs[30][2] == 0x30300002u &&
         state.vgprs[30][3] == 0x30300003u &&
         state.vgprs[31][0] == 0x12340001u &&
         state.vgprs[31][1] == 0x12340002u &&
         state.vgprs[31][2] == 0x31313131u &&
         state.vgprs[31][3] == 0x12340004u &&
         state.vgprs[41][0] == 0x12340001u &&
         state.vgprs[41][1] == 0x12340002u &&
         state.vgprs[41][2] == 0x41414141u &&
         state.vgprs[41][3] == 0x12340004u &&
         state.vgprs[53][0] == 0x98760001u &&
         state.vgprs[53][1] == 0x98760002u &&
         state.vgprs[53][2] == 0x53535353u &&
         state.vgprs[53][3] == 0x98760004u &&
         state.vgprs[61][0] == 0x98760001u &&
         state.vgprs[61][1] == 0x98760002u &&
         state.vgprs[61][2] == 0x61616161u &&
         state.vgprs[61][3] == 0x98760004u &&
         state.vgprs[73][0] == 0x22220001u &&
         state.vgprs[73][1] == 0x22220002u &&
         state.vgprs[73][2] == 0x73737373u &&
         state.vgprs[73][3] == 0x22220004u &&
         state.vgprs[81][0] == 0x11110001u &&
         state.vgprs[81][1] == 0x11110002u &&
         state.vgprs[81][2] == 0x81818181u &&
         state.vgprs[81][3] == 0x11110004u &&
         state.halted && !state.waiting_on_barrier && state.pc == 5u;
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

bool ExpectUnaryMathSeedState(const mirage::sim::isa::WaveExecutionState& state) {
  std::uint32_t source_f64_low = 0;
  std::uint32_t source_f64_high = 0;
  SplitU64(DoubleBits(4.0), &source_f64_low, &source_f64_high);
  std::uint32_t rcp_f64_low = 0;
  std::uint32_t rcp_f64_high = 0;
  SplitU64(DoubleBits(0.25), &rcp_f64_low, &rcp_f64_high);
  std::uint32_t rsq_f64_low = 0;
  std::uint32_t rsq_f64_high = 0;
  SplitU64(DoubleBits(0.5), &rsq_f64_low, &rsq_f64_high);
  std::uint32_t sqrt_f64_low = 0;
  std::uint32_t sqrt_f64_high = 0;
  SplitU64(DoubleBits(2.0), &sqrt_f64_low, &sqrt_f64_high);

  return state.vgprs[10][0] == FloatBits(0.25f) &&
         state.vgprs[10][1] == FloatBits(0.25f) &&
         state.vgprs[10][2] == 0x10101010u &&
         state.vgprs[10][3] == FloatBits(0.25f) &&
         state.vgprs[11][0] == FloatBits(0.25f) &&
         state.vgprs[11][1] == FloatBits(0.25f) &&
         state.vgprs[11][2] == 0x11111111u &&
         state.vgprs[11][3] == FloatBits(0.25f) &&
         state.vgprs[12][0] == FloatBits(0.5f) &&
         state.vgprs[12][1] == FloatBits(0.5f) &&
         state.vgprs[12][2] == 0x12121212u &&
         state.vgprs[12][3] == FloatBits(0.5f) &&
         state.vgprs[13][0] == FloatBits(2.0f) &&
         state.vgprs[13][1] == FloatBits(2.0f) &&
         state.vgprs[13][2] == 0x13131313u &&
         state.vgprs[13][3] == FloatBits(2.0f) &&
         state.vgprs[14][0] == FloatBits(8.0f) &&
         state.vgprs[14][1] == FloatBits(8.0f) &&
         state.vgprs[14][2] == 0x14141414u &&
         state.vgprs[14][3] == FloatBits(8.0f) &&
         state.vgprs[15][0] == FloatBits(3.0f) &&
         state.vgprs[15][1] == FloatBits(3.0f) &&
         state.vgprs[15][2] == 0x15151515u &&
         state.vgprs[15][3] == FloatBits(3.0f) &&
         state.vgprs[16][0] == FloatBits(0.0f) &&
         state.vgprs[16][1] == FloatBits(0.0f) &&
         state.vgprs[16][2] == 0x16161616u &&
         state.vgprs[16][3] == FloatBits(0.0f) &&
         state.vgprs[17][0] == FloatBits(1.0f) &&
         state.vgprs[17][1] == FloatBits(1.0f) &&
         state.vgprs[17][2] == 0x17171717u &&
         state.vgprs[17][3] == FloatBits(1.0f) &&
         state.vgprs[20][0] == source_f64_low &&
         state.vgprs[20][1] == source_f64_low &&
         state.vgprs[20][2] == 0x20202020u &&
         state.vgprs[20][3] == source_f64_low &&
         state.vgprs[21][0] == source_f64_high &&
         state.vgprs[21][1] == source_f64_high &&
         state.vgprs[21][2] == 0x21212121u &&
         state.vgprs[21][3] == source_f64_high &&
         state.vgprs[30][0] == rcp_f64_low &&
         state.vgprs[30][1] == rcp_f64_low &&
         state.vgprs[30][2] == 0x30303030u &&
         state.vgprs[30][3] == rcp_f64_low &&
         state.vgprs[31][0] == rcp_f64_high &&
         state.vgprs[31][1] == rcp_f64_high &&
         state.vgprs[31][2] == 0x31313131u &&
         state.vgprs[31][3] == rcp_f64_high &&
         state.vgprs[32][0] == rsq_f64_low &&
         state.vgprs[32][1] == rsq_f64_low &&
         state.vgprs[32][2] == 0x32323232u &&
         state.vgprs[32][3] == rsq_f64_low &&
         state.vgprs[33][0] == rsq_f64_high &&
         state.vgprs[33][1] == rsq_f64_high &&
         state.vgprs[33][2] == 0x33333333u &&
         state.vgprs[33][3] == rsq_f64_high &&
         state.vgprs[34][0] == sqrt_f64_low &&
         state.vgprs[34][1] == sqrt_f64_low &&
         state.vgprs[34][2] == 0x34343434u &&
         state.vgprs[34][3] == sqrt_f64_low &&
         state.vgprs[35][0] == sqrt_f64_high &&
         state.vgprs[35][1] == sqrt_f64_high &&
         state.vgprs[35][2] == 0x35353535u &&
         state.vgprs[35][3] == sqrt_f64_high && state.exec_mask == 0xbu &&
         state.halted && !state.waiting_on_barrier && state.pc == 16u;
}

bool ExpectUnaryCountConvertSeedState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.vgprs[10][0] == 16u && state.vgprs[10][1] == 0xffffffffu &&
         state.vgprs[10][2] == 0x10101010u && state.vgprs[10][3] == 31u &&
         state.vgprs[11][0] == 12u && state.vgprs[11][1] == 0xffffffffu &&
         state.vgprs[11][2] == 0x11111111u && state.vgprs[11][3] == 0u &&
         state.vgprs[12][0] == 16u && state.vgprs[12][1] == 0xffffffffu &&
         state.vgprs[12][2] == 0x12121212u && state.vgprs[12][3] == 1u &&
         state.vgprs[13][0] == 0xffffffffu && state.vgprs[13][1] == 2u &&
         state.vgprs[13][2] == 0x13131313u && state.vgprs[13][3] == 4u &&
         state.vgprs[14][0] == 0xfffffffeu && state.vgprs[14][1] == 2u &&
         state.vgprs[14][2] == 0x14141414u && state.vgprs[14][3] == 3u &&
         state.vgprs[15][0] == 0xfffffffdu && state.vgprs[15][1] == 2u &&
         state.vgprs[15][2] == 0x15151515u &&
         state.vgprs[15][3] == 0xffff8001u &&
         state.vgprs[16][0] == 0x0000fffdu && state.vgprs[16][1] == 2u &&
         state.vgprs[16][2] == 0x16161616u &&
         state.vgprs[16][3] == 0x00008001u &&
         state.exec_mask == 0xbu && state.halted &&
         !state.waiting_on_barrier && state.pc == 7u;
}

bool ExpectF16BridgeSeedState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.vgprs[40][0] == 0x00003e00u &&
         state.vgprs[40][1] == 0x0000c000u &&
         state.vgprs[40][2] == 0x40404040u &&
         state.vgprs[40][3] == 0x00003800u &&
         state.vgprs[41][0] == 0x0000c000u &&
         state.vgprs[41][1] == 0x00004200u &&
         state.vgprs[41][2] == 0x41414141u &&
         state.vgprs[41][3] == 0x00000000u &&
         state.vgprs[42][0] == 0x00004000u &&
         state.vgprs[42][1] == 0x00004700u &&
         state.vgprs[42][2] == 0x42424242u &&
         state.vgprs[42][3] == 0x00003c00u &&
         state.vgprs[43][0] == FloatBits(1.5f) &&
         state.vgprs[43][1] == FloatBits(-2.0f) &&
         state.vgprs[43][2] == 0x43434343u &&
         state.vgprs[43][3] == FloatBits(0.5f) &&
         state.vgprs[44][0] == FloatBits(-2.0f) &&
         state.vgprs[44][1] == FloatBits(3.0f) &&
         state.vgprs[44][2] == 0x44444444u &&
         state.vgprs[44][3] == FloatBits(0.0f) &&
         state.vgprs[45][0] == FloatBits(2.0f) &&
         state.vgprs[45][1] == FloatBits(7.0f) &&
         state.vgprs[45][2] == 0x45454545u &&
         state.vgprs[45][3] == FloatBits(1.0f) &&
         state.vgprs[46][0] == 0x00000001u &&
         state.vgprs[46][1] == 0x0000fffeu &&
         state.vgprs[46][2] == 0x46464646u &&
         state.vgprs[46][3] == 0x00000000u &&
         state.vgprs[47][0] == 0x00000002u &&
         state.vgprs[47][1] == 0x00000007u &&
         state.vgprs[47][2] == 0x47474747u &&
         state.vgprs[47][3] == 0x00000001u &&
         state.exec_mask == 0xbu && state.halted &&
         !state.waiting_on_barrier && state.pc == 9u;
}

bool ExpectHalfConsumerSeedState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.vgprs[48][0] == 0x00001234u &&
         state.vgprs[48][1] == 0x00008001u &&
         state.vgprs[48][2] == 0x48484848u &&
         state.vgprs[48][3] == 0x00000000u &&
         state.vgprs[49][0] == 0x00004110u &&
         state.vgprs[49][1] == 0x00004110u &&
         state.vgprs[49][2] == 0x49494949u &&
         state.vgprs[49][3] == 0x00004110u &&
         state.vgprs[50][0] == 0x11223344u &&
         state.vgprs[50][1] == 0x55667788u &&
         state.vgprs[50][2] == 0x50505050u &&
         state.vgprs[50][3] == 0x99aabbccu &&
         state.exec_mask == 0xbu && state.halted &&
         !state.waiting_on_barrier && state.pc == 3u;
}

bool ExpectSwapSeedState(const mirage::sim::isa::WaveExecutionState& state) {
  return state.vgprs[60][0] == 0xaaaaaaaau &&
         state.vgprs[60][1] == 0xbbbbbbbbu &&
         state.vgprs[60][2] == 0x60606060u &&
         state.vgprs[60][3] == 0xccccccccu &&
         state.vgprs[61][0] == 0x11111111u &&
         state.vgprs[61][1] == 0x22222222u &&
         state.vgprs[61][2] == 0x61616161u &&
         state.vgprs[61][3] == 0x33333333u &&
         state.vgprs[62][0] == 0x0000beefu &&
         state.vgprs[62][1] == 0x00000002u &&
         state.vgprs[62][2] == 0x62626262u &&
         state.vgprs[62][3] == 0x00001234u &&
         state.vgprs[63][0] == 0x00001234u &&
         state.vgprs[63][1] == 0x00008001u &&
         state.vgprs[63][2] == 0x63636363u &&
         state.vgprs[63][3] == 0x0000ffffu &&
         state.exec_mask == 0xbu && state.halted &&
         !state.waiting_on_barrier && state.pc == 2u;
}

bool ExpectOffsetNormSeedState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.vgprs[70][0] == FloatBits(-0.5f) &&
         state.vgprs[70][1] == FloatBits(0.125f) &&
         state.vgprs[70][2] == 0x70707070u &&
         state.vgprs[70][3] == FloatBits(-0.0625f) &&
         state.vgprs[71][0] == 0x00004000u &&
         state.vgprs[71][1] == 0x0000c000u &&
         state.vgprs[71][2] == 0x71717171u &&
         state.vgprs[71][3] == 0x00002000u &&
         state.vgprs[72][0] == 0x00004000u &&
         state.vgprs[72][1] == 0x0000ffffu &&
         state.vgprs[72][2] == 0x72727272u &&
         state.vgprs[72][3] == 0x00000000u &&
         state.exec_mask == 0xbu && state.halted &&
         !state.waiting_on_barrier && state.pc == 4u;
}

bool ExpectF16UnaryMathSeedState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.vgprs[80][0] == 0x00003400u &&
         state.vgprs[80][1] == 0x00003400u &&
         state.vgprs[80][2] == 0x80808080u &&
         state.vgprs[80][3] == 0x00003400u &&
         state.vgprs[81][0] == 0x00003800u &&
         state.vgprs[81][1] == 0x00003800u &&
         state.vgprs[81][2] == 0x81818181u &&
         state.vgprs[81][3] == 0x00003800u &&
         state.vgprs[82][0] == 0x00004000u &&
         state.vgprs[82][1] == 0x00004000u &&
         state.vgprs[82][2] == 0x82828282u &&
         state.vgprs[82][3] == 0x00004000u &&
         state.vgprs[83][0] == 0x00004800u &&
         state.vgprs[83][1] == 0x00004800u &&
         state.vgprs[83][2] == 0x83838383u &&
         state.vgprs[83][3] == 0x00004800u &&
         state.vgprs[84][0] == 0x00004200u &&
         state.vgprs[84][1] == 0x00004200u &&
         state.vgprs[84][2] == 0x84848484u &&
         state.vgprs[84][3] == 0x00004200u &&
         state.vgprs[85][0] == 0x00000000u &&
         state.vgprs[85][1] == 0x00000000u &&
         state.vgprs[85][2] == 0x85858585u &&
         state.vgprs[85][3] == 0x00000000u &&
         state.vgprs[86][0] == 0x00003c00u &&
         state.vgprs[86][1] == 0x00003c00u &&
         state.vgprs[86][2] == 0x86868686u &&
         state.vgprs[86][3] == 0x00003c00u &&
         state.vgprs[87][0] == 0x00000003u &&
         state.vgprs[87][1] == 0x00000003u &&
         state.vgprs[87][2] == 0x87878787u &&
         state.vgprs[87][3] == 0x00000003u &&
         state.vgprs[88][0] == 0x00003800u &&
         state.vgprs[88][1] == 0x00003800u &&
         state.vgprs[88][2] == 0x88888888u &&
         state.vgprs[88][3] == 0x00003800u &&
         state.vgprs[89][0] == 0x00003400u &&
         state.vgprs[89][1] == 0x00003400u &&
         state.vgprs[89][2] == 0x89898989u &&
         state.vgprs[89][3] == 0x00003400u &&
         state.vgprs[90][0] == 0x0000c000u &&
         state.vgprs[90][1] == 0x0000c000u &&
         state.vgprs[90][2] == 0x90909090u &&
         state.vgprs[90][3] == 0x0000c000u &&
         state.vgprs[91][0] == 0x0000c000u &&
         state.vgprs[91][1] == 0x0000c000u &&
         state.vgprs[91][2] == 0x91919191u &&
         state.vgprs[91][3] == 0x0000c000u &&
         state.vgprs[92][0] == 0x0000c200u &&
         state.vgprs[92][1] == 0x0000c200u &&
         state.vgprs[92][2] == 0x92929292u &&
         state.vgprs[92][3] == 0x0000c200u &&
         state.vgprs[93][0] == 0x0000c200u &&
         state.vgprs[93][1] == 0x0000c200u &&
         state.vgprs[93][2] == 0x93939393u &&
         state.vgprs[93][3] == 0x0000c200u &&
         state.exec_mask == 0xbu && state.halted &&
         !state.waiting_on_barrier && state.pc == 14u;
}

bool ExpectF16VectorBinarySeedState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.vgprs[94][0] == 0x00004000u &&
         state.vgprs[94][1] == 0x00004000u &&
         state.vgprs[94][2] == 0x94949494u &&
         state.vgprs[94][3] == 0x00004000u &&
         state.vgprs[95][0] == 0x00003c00u &&
         state.vgprs[95][1] == 0x00003c00u &&
         state.vgprs[95][2] == 0x95959595u &&
         state.vgprs[95][3] == 0x00003c00u &&
         state.vgprs[96][0] == 0x0000bc00u &&
         state.vgprs[96][1] == 0x0000bc00u &&
         state.vgprs[96][2] == 0x96969696u &&
         state.vgprs[96][3] == 0x0000bc00u &&
         state.vgprs[97][0] == 0x00003800u &&
         state.vgprs[97][1] == 0x00003800u &&
         state.vgprs[97][2] == 0x97979797u &&
         state.vgprs[97][3] == 0x00003800u &&
         state.vgprs[98][0] == 0x00003e00u &&
         state.vgprs[98][1] == 0x00003e00u &&
         state.vgprs[98][2] == 0x98989898u &&
         state.vgprs[98][3] == 0x00003e00u &&
         state.exec_mask == 0xbu && state.halted &&
         !state.waiting_on_barrier && state.pc == 5u;
}

bool ExpectHalfPackMulSeedState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.vgprs[99][0] == 0x0000ff00u &&
         state.vgprs[99][1] == 0x00006405u &&
         state.vgprs[99][2] == 0x99999999u &&
         state.vgprs[99][3] == 0x000000ffu &&
         state.vgprs[100][0] == 0x00004200u &&
         state.vgprs[100][1] == 0x00004000u &&
         state.vgprs[100][2] == 0xa0a0a0a0u &&
         state.vgprs[100][3] == 0x0000be00u &&
         state.exec_mask == 0xbu && state.halted &&
         !state.waiting_on_barrier && state.pc == 2u;
}

bool ExpectHalfPackExponentSeedState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.vgprs[101][0] == 0xbc003c00u &&
         state.vgprs[101][1] == 0x34004100u &&
         state.vgprs[101][2] == 0xa1a1a1a1u &&
         state.vgprs[101][3] == 0x3e00ba00u &&
         state.vgprs[102][0] == 0x00004000u &&
         state.vgprs[102][1] == 0x0000b400u &&
         state.vgprs[102][2] == 0xa2a2a2a2u &&
         state.vgprs[102][3] == 0x00004200u &&
         state.exec_mask == 0xbu && state.halted &&
         !state.waiting_on_barrier && state.pc == 2u;
}

bool ExpectF32VectorBinarySeedState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.vgprs[103][0] == FloatBits(2.0f) &&
         state.vgprs[103][1] == FloatBits(1.0f) &&
         state.vgprs[103][2] == 0xa3a3a3a3u &&
         state.vgprs[103][3] == FloatBits(3.0f) &&
         state.vgprs[104][0] == FloatBits(1.0f) &&
         state.vgprs[104][1] == FloatBits(-5.0f) &&
         state.vgprs[104][2] == 0xa4a4a4a4u &&
         state.vgprs[104][3] == FloatBits(5.0f) &&
         state.vgprs[105][0] == FloatBits(-1.0f) &&
         state.vgprs[105][1] == FloatBits(5.0f) &&
         state.vgprs[105][2] == 0xa5a5a5a5u &&
         state.vgprs[105][3] == FloatBits(-5.0f) &&
         state.vgprs[106][0] == FloatBits(0.75f) &&
         state.vgprs[106][1] == FloatBits(-6.0f) &&
         state.vgprs[106][2] == 0xa6a6a6a6u &&
         state.vgprs[106][3] == FloatBits(-4.0f) &&
         state.vgprs[107][0] == FloatBits(0.5f) &&
         state.vgprs[107][1] == FloatBits(-2.0f) &&
         state.vgprs[107][2] == 0xa7a7a7a7u &&
         state.vgprs[107][3] == FloatBits(-1.0f) &&
         state.vgprs[108][0] == FloatBits(1.5f) &&
         state.vgprs[108][1] == FloatBits(3.0f) &&
         state.vgprs[108][2] == 0xa8a8a8a8u &&
         state.vgprs[108][3] == FloatBits(4.0f) &&
         state.exec_mask == 0xbu && state.halted &&
         !state.waiting_on_barrier && state.pc == 6u;
}

bool ExpectDx9FmacSeedState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.vgprs[118][0] == 0u &&
         state.vgprs[118][1] == 0u &&
         state.vgprs[118][2] == 0xc9c9c9c9u &&
         state.vgprs[118][3] == FloatBits(-8.0f) &&
         state.vgprs[119][0] == FloatBits(7.0f) &&
         state.vgprs[119][1] == FloatBits(-3.0f) &&
         state.vgprs[119][2] == 0xd0d0d0d0u &&
         state.vgprs[119][3] == FloatBits(7.0f) &&
         state.vgprs[120][0] == 0x00004000u &&
         state.vgprs[120][1] == 0x0000b800u &&
         state.vgprs[120][2] == 0xd1d1d1d1u &&
         state.vgprs[120][3] == 0x00004100u &&
         state.exec_mask == 0xbu && state.halted &&
         !state.waiting_on_barrier && state.pc == 3u;
}

bool RunDx9FmacBatchTest(
    const mirage::sim::isa::Gfx1201BinaryDecoder& decoder,
    const mirage::sim::isa::Gfx1201Interpreter& interpreter,
    std::string* error_message) {
  using namespace mirage::sim::isa;

  const std::array<std::uint32_t, 4> dx9_fmac_words{
      MakeVop2(7u, 118u, 257u, 2u),
      MakeVop2(43u, 119u, 259u, 4u),
      MakeVop2(54u, 120u, 261u, 6u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> dx9_fmac_program;
  if (!Expect(decoder.DecodeProgram(dx9_fmac_words, &dx9_fmac_program,
                                    error_message),
              "expected DX9/FMAC program decode success") ||
      !Expect(dx9_fmac_program.size() == 4u,
              "expected four decoded DX9/FMAC instructions") ||
      !Expect(dx9_fmac_program[0].opcode == "V_MUL_DX9_ZERO_F32",
              "expected decoded V_MUL_DX9_ZERO_F32") ||
      !Expect(dx9_fmac_program[1].opcode == "V_FMAC_F32",
              "expected decoded V_FMAC_F32") ||
      !Expect(dx9_fmac_program[2].opcode == "V_FMAC_F16",
              "expected decoded V_FMAC_F16") ||
      !Expect(dx9_fmac_program[3].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after DX9/FMAC batch")) {
    return false;
  }

  auto initialize_dx9_fmac_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;

    state->vgprs[1][0] = 0u;
    state->vgprs[1][1] = 0x80000000u;
    state->vgprs[1][2] = 0x11111111u;
    state->vgprs[1][3] = FloatBits(2.0f);

    state->vgprs[2][0] = 0x7f800000u;
    state->vgprs[2][1] = kQuietNaNF32Bits;
    state->vgprs[2][2] = 0x22222222u;
    state->vgprs[2][3] = FloatBits(-4.0f);

    state->vgprs[119][0] = FloatBits(1.0f);
    state->vgprs[119][1] = FloatBits(-1.0f);
    state->vgprs[119][2] = 0xd0d0d0d0u;
    state->vgprs[119][3] = FloatBits(10.0f);

    state->vgprs[3][0] = FloatBits(2.0f);
    state->vgprs[3][1] = FloatBits(4.0f);
    state->vgprs[3][2] = 0x33333333u;
    state->vgprs[3][3] = FloatBits(-2.0f);

    state->vgprs[4][0] = FloatBits(3.0f);
    state->vgprs[4][1] = FloatBits(-0.5f);
    state->vgprs[4][2] = 0x44444444u;
    state->vgprs[4][3] = FloatBits(1.5f);

    state->vgprs[120][0] = 0x00003c00u;
    state->vgprs[120][1] = 0x0000bc00u;
    state->vgprs[120][2] = 0xd1d1d1d1u;
    state->vgprs[120][3] = 0x00003800u;

    state->vgprs[5][0] = 0x00003800u;
    state->vgprs[5][1] = 0x00003c00u;
    state->vgprs[5][2] = 0x55555555u;
    state->vgprs[5][3] = 0x0000c000u;

    state->vgprs[6][0] = 0x00004000u;
    state->vgprs[6][1] = 0x00003800u;
    state->vgprs[6][2] = 0x66666666u;
    state->vgprs[6][3] = 0x0000bc00u;

    state->vgprs[118][2] = 0xc9c9c9c9u;
  };

  WaveExecutionState decoded_dx9_fmac_state;
  initialize_dx9_fmac_state(&decoded_dx9_fmac_state);
  if (!Expect(interpreter.ExecuteProgram(dx9_fmac_program,
                                         &decoded_dx9_fmac_state,
                                         error_message),
              "expected decoded DX9/FMAC execution success") ||
      !Expect(ExpectDx9FmacSeedState(decoded_dx9_fmac_state),
              "expected decoded DX9/FMAC state")) {
    return false;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_dx9_fmac_program;
  if (!Expect(interpreter.CompileProgram(dx9_fmac_program,
                                         &compiled_dx9_fmac_program,
                                         error_message),
              "expected compiled DX9/FMAC program success") ||
      !Expect(compiled_dx9_fmac_program.size() == 4u,
              "expected four compiled DX9/FMAC instructions") ||
      !Expect(compiled_dx9_fmac_program[0].opcode ==
                  Gfx1201CompiledOpcode::kVMulDx9ZeroF32,
              "expected compiled V_MUL_DX9_ZERO_F32 opcode") ||
      !Expect(compiled_dx9_fmac_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVFmacF32,
              "expected compiled V_FMAC_F32 opcode") ||
      !Expect(compiled_dx9_fmac_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVFmacF16,
              "expected compiled V_FMAC_F16 opcode")) {
    return false;
  }

  WaveExecutionState compiled_dx9_fmac_state;
  initialize_dx9_fmac_state(&compiled_dx9_fmac_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_dx9_fmac_program,
                                         &compiled_dx9_fmac_state,
                                         error_message),
              "expected compiled DX9/FMAC execution success") ||
      !Expect(ExpectDx9FmacSeedState(compiled_dx9_fmac_state),
              "expected compiled DX9/FMAC state")) {
    return false;
  }

  return true;
}

bool ExpectPackedFmacSeedState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.vgprs[121][0] == 0xc0004000u &&
         state.vgprs[121][1] == 0x40004100u &&
         state.vgprs[121][2] == 0xc2c2c2c2u &&
         state.vgprs[121][3] == 0x41003800u &&
         state.exec_mask == 0xbu && state.halted &&
         !state.waiting_on_barrier && state.pc == 1u;
}

bool RunPackedFmacBatchTest(
    const mirage::sim::isa::Gfx1201BinaryDecoder& decoder,
    const mirage::sim::isa::Gfx1201Interpreter& interpreter,
    std::string* error_message) {
  using namespace mirage::sim::isa;

  const std::array<std::uint32_t, 2> packed_fmac_words{
      MakeVop2(60u, 121u, 257u, 2u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> packed_fmac_program;
  if (!Expect(decoder.DecodeProgram(packed_fmac_words, &packed_fmac_program,
                                    error_message),
              "expected packed FMAC program decode success") ||
      !Expect(packed_fmac_program.size() == 2u,
              "expected two decoded packed FMAC instructions") ||
      !Expect(packed_fmac_program[0].opcode == "V_PK_FMAC_F16",
              "expected decoded V_PK_FMAC_F16") ||
      !Expect(packed_fmac_program[1].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after packed FMAC batch")) {
    return false;
  }

  auto initialize_packed_fmac_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;

    state->vgprs[121][0] = 0xbc003c00u;
    state->vgprs[121][1] = 0x3c003800u;
    state->vgprs[121][2] = 0xc2c2c2c2u;
    state->vgprs[121][3] = 0x3800b800u;

    state->vgprs[1][0] = 0x40003800u;
    state->vgprs[1][1] = 0x3c00c000u;
    state->vgprs[1][2] = 0x11111111u;
    state->vgprs[1][3] = 0xbc003c00u;

    state->vgprs[2][0] = 0xb8004000u;
    state->vgprs[2][1] = 0x3c00bc00u;
    state->vgprs[2][2] = 0x22222222u;
    state->vgprs[2][3] = 0xc0003c00u;
  };

  WaveExecutionState decoded_packed_fmac_state;
  initialize_packed_fmac_state(&decoded_packed_fmac_state);
  if (!Expect(interpreter.ExecuteProgram(packed_fmac_program,
                                         &decoded_packed_fmac_state,
                                         error_message),
              "expected decoded packed FMAC execution success") ||
      !Expect(ExpectPackedFmacSeedState(decoded_packed_fmac_state),
              "expected decoded packed FMAC state")) {
    return false;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_packed_fmac_program;
  if (!Expect(interpreter.CompileProgram(packed_fmac_program,
                                         &compiled_packed_fmac_program,
                                         error_message),
              "expected compiled packed FMAC program success") ||
      !Expect(compiled_packed_fmac_program.size() == 2u,
              "expected two compiled packed FMAC instructions") ||
      !Expect(compiled_packed_fmac_program[0].opcode ==
                  Gfx1201CompiledOpcode::kVPkFmacF16,
              "expected compiled V_PK_FMAC_F16 opcode")) {
    return false;
  }

  WaveExecutionState compiled_packed_fmac_state;
  initialize_packed_fmac_state(&compiled_packed_fmac_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_packed_fmac_program,
                                         &compiled_packed_fmac_state,
                                         error_message),
              "expected compiled packed FMAC execution success") ||
      !Expect(ExpectPackedFmacSeedState(compiled_packed_fmac_state),
              "expected compiled packed FMAC state")) {
    return false;
  }

  return true;
}

bool ExpectHalfLiteralFmaSeedState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.vgprs[125][0] == 0x00004200u &&
         state.vgprs[125][1] == 0x0000bc00u &&
         state.vgprs[125][2] == 0xa5a5a5a5u &&
         state.vgprs[125][3] == 0u &&
         state.vgprs[126][0] == 0x00004400u &&
         state.vgprs[126][1] == 0x00003c00u &&
         state.vgprs[126][2] == 0xb6b6b6b6u &&
         state.vgprs[126][3] == 0x00003e00u &&
         state.exec_mask == 0xbu && state.halted &&
         !state.waiting_on_barrier && state.pc == 2u;
}

bool RunHalfLiteralFmaBatchTest(
    const mirage::sim::isa::Gfx1201BinaryDecoder& decoder,
    const mirage::sim::isa::Gfx1201Interpreter& interpreter,
    std::string* error_message) {
  using namespace mirage::sim::isa;

  const std::array<std::uint32_t, 5> half_literal_fma_words{
      MakeVop2(55u, 125u, 257u, 2u),
      0x00003c00u,
      MakeVop2(56u, 126u, 257u, 3u),
      0x00004000u,
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> half_literal_fma_program;
  if (!Expect(decoder.DecodeProgram(half_literal_fma_words,
                                    &half_literal_fma_program, error_message),
              "expected half literal FMA program decode success") ||
      !Expect(half_literal_fma_program.size() == 3u,
              "expected three decoded half literal FMA instructions") ||
      !Expect(half_literal_fma_program[0].opcode == "V_FMAMK_F16",
              "expected decoded V_FMAMK_F16") ||
      !Expect(half_literal_fma_program[1].opcode == "V_FMAAK_F16",
              "expected decoded V_FMAAK_F16") ||
      !Expect(half_literal_fma_program[2].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after half literal FMA batch")) {
    return false;
  }

  auto initialize_half_literal_fma_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;

    state->vgprs[1][0] = 0x00003c00u;
    state->vgprs[1][1] = 0x0000c000u;
    state->vgprs[1][2] = 0x11111111u;
    state->vgprs[1][3] = 0x00003800u;

    state->vgprs[2][0] = 0x00004000u;
    state->vgprs[2][1] = 0x00003c00u;
    state->vgprs[2][2] = 0x22222222u;
    state->vgprs[2][3] = 0x0000b800u;

    state->vgprs[3][0] = 0x00004000u;
    state->vgprs[3][1] = 0x00003800u;
    state->vgprs[3][2] = 0x33333333u;
    state->vgprs[3][3] = 0x0000bc00u;

    state->vgprs[125][2] = 0xa5a5a5a5u;
    state->vgprs[126][2] = 0xb6b6b6b6u;
  };

  WaveExecutionState decoded_half_literal_fma_state;
  initialize_half_literal_fma_state(&decoded_half_literal_fma_state);
  if (!Expect(interpreter.ExecuteProgram(half_literal_fma_program,
                                         &decoded_half_literal_fma_state,
                                         error_message),
              "expected decoded half literal FMA execution success") ||
      !Expect(ExpectHalfLiteralFmaSeedState(decoded_half_literal_fma_state),
              "expected decoded half literal FMA state")) {
    return false;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_half_literal_fma_program;
  if (!Expect(interpreter.CompileProgram(half_literal_fma_program,
                                         &compiled_half_literal_fma_program,
                                         error_message),
              "expected compiled half literal FMA program success") ||
      !Expect(compiled_half_literal_fma_program.size() == 3u,
              "expected three compiled half literal FMA instructions") ||
      !Expect(compiled_half_literal_fma_program[0].opcode ==
                  Gfx1201CompiledOpcode::kVFmamkF16,
              "expected compiled V_FMAMK_F16 opcode") ||
      !Expect(compiled_half_literal_fma_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVFmaakF16,
              "expected compiled V_FMAAK_F16 opcode")) {
    return false;
  }

  WaveExecutionState compiled_half_literal_fma_state;
  initialize_half_literal_fma_state(&compiled_half_literal_fma_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_half_literal_fma_program,
                                         &compiled_half_literal_fma_state,
                                         error_message),
              "expected compiled half literal FMA execution success") ||
      !Expect(ExpectHalfLiteralFmaSeedState(compiled_half_literal_fma_state),
              "expected compiled half literal FMA state")) {
    return false;
  }

  return true;
}

bool ExpectCarryChainSeedState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.vgprs[122][0] == 0u &&
         state.vgprs[122][1] == 13u &&
         state.vgprs[122][2] == 0xaaaaaaaau &&
         state.vgprs[122][3] == 0xffffffffu &&
         state.vgprs[123][0] == 6u &&
         state.vgprs[123][1] == 0xfffffffdu &&
         state.vgprs[123][2] == 0xbbbbbbbbu &&
         state.vgprs[123][3] == 0u &&
         state.vgprs[124][0] == 5u &&
         state.vgprs[124][1] == 0xfffffffbu &&
         state.vgprs[124][2] == 0xccccccccu &&
         state.vgprs[124][3] == 0xffffffffu &&
         state.vcc_mask == 0x5u && state.exec_mask == 0xbu && state.halted &&
         !state.waiting_on_barrier && state.pc == 3u;
}

bool RunCarryChainBatchTest(
    const mirage::sim::isa::Gfx1201BinaryDecoder& decoder,
    const mirage::sim::isa::Gfx1201Interpreter& interpreter,
    std::string* error_message) {
  using namespace mirage::sim::isa;

  const std::array<std::uint32_t, 4> carry_chain_words{
      MakeVop2(32u, 122u, 257u, 2u),
      MakeVop2(33u, 123u, 259u, 4u),
      MakeVop2(34u, 124u, 261u, 6u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> carry_chain_program;
  if (!Expect(decoder.DecodeProgram(carry_chain_words, &carry_chain_program,
                                    error_message),
              "expected carry-chain program decode success") ||
      !Expect(carry_chain_program.size() == 4u,
              "expected four decoded carry-chain instructions") ||
      !Expect(carry_chain_program[0].opcode == "V_ADD_CO_CI_U32",
              "expected decoded V_ADD_CO_CI_U32") ||
      !Expect(carry_chain_program[1].opcode == "V_SUB_CO_CI_U32",
              "expected decoded V_SUB_CO_CI_U32") ||
      !Expect(carry_chain_program[2].opcode == "V_SUBREV_CO_CI_U32",
              "expected decoded V_SUBREV_CO_CI_U32") ||
      !Expect(carry_chain_program[3].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after carry-chain batch")) {
    return false;
  }

  auto initialize_carry_chain_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;
    state->vcc_mask = 0xeu;

    state->vgprs[122][2] = 0xaaaaaaaau;
    state->vgprs[123][2] = 0xbbbbbbbbu;
    state->vgprs[124][2] = 0xccccccccu;

    state->vgprs[1][0] = 0xffffffffu;
    state->vgprs[1][1] = 5u;
    state->vgprs[1][2] = 0x11111111u;
    state->vgprs[1][3] = 0xfffffffeu;

    state->vgprs[2][0] = 1u;
    state->vgprs[2][1] = 7u;
    state->vgprs[2][2] = 0x22222222u;
    state->vgprs[2][3] = 0u;

    state->vgprs[3][0] = 10u;
    state->vgprs[3][1] = 2u;
    state->vgprs[3][2] = 0x33333333u;
    state->vgprs[3][3] = 0u;

    state->vgprs[4][0] = 3u;
    state->vgprs[4][1] = 5u;
    state->vgprs[4][2] = 0x44444444u;
    state->vgprs[4][3] = 0u;

    state->vgprs[5][0] = 4u;
    state->vgprs[5][1] = 8u;
    state->vgprs[5][2] = 0x55555555u;
    state->vgprs[5][3] = 0u;

    state->vgprs[6][0] = 10u;
    state->vgprs[6][1] = 3u;
    state->vgprs[6][2] = 0x66666666u;
    state->vgprs[6][3] = 0u;
  };

  WaveExecutionState decoded_carry_chain_state;
  initialize_carry_chain_state(&decoded_carry_chain_state);
  if (!Expect(interpreter.ExecuteProgram(carry_chain_program,
                                         &decoded_carry_chain_state,
                                         error_message),
              "expected decoded carry-chain execution success") ||
      !Expect(ExpectCarryChainSeedState(decoded_carry_chain_state),
              "expected decoded carry-chain state")) {
    return false;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_carry_chain_program;
  if (!Expect(interpreter.CompileProgram(carry_chain_program,
                                         &compiled_carry_chain_program,
                                         error_message),
              "expected compiled carry-chain program success") ||
      !Expect(compiled_carry_chain_program.size() == 4u,
              "expected four compiled carry-chain instructions") ||
      !Expect(compiled_carry_chain_program[0].opcode ==
                  Gfx1201CompiledOpcode::kVAddCoCiU32,
              "expected compiled V_ADD_CO_CI_U32 opcode") ||
      !Expect(compiled_carry_chain_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVSubCoCiU32,
              "expected compiled V_SUB_CO_CI_U32 opcode") ||
      !Expect(compiled_carry_chain_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVSubrevCoCiU32,
              "expected compiled V_SUBREV_CO_CI_U32 opcode")) {
    return false;
  }

  WaveExecutionState compiled_carry_chain_state;
  initialize_carry_chain_state(&compiled_carry_chain_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_carry_chain_program,
                                         &compiled_carry_chain_state,
                                         error_message),
              "expected compiled carry-chain execution success") ||
      !Expect(ExpectCarryChainSeedState(compiled_carry_chain_state),
              "expected compiled carry-chain state")) {
    return false;
  }

  return true;
}

bool ExpectF64VectorBinarySeedState(
    const mirage::sim::isa::WaveExecutionState& state) {
  std::uint32_t add_lane0_low, add_lane0_high;
  std::uint32_t add_lane1_low, add_lane1_high;
  std::uint32_t add_lane3_low, add_lane3_high;
  std::uint32_t mul_lane0_low, mul_lane0_high;
  std::uint32_t mul_lane1_low, mul_lane1_high;
  std::uint32_t mul_lane3_low, mul_lane3_high;
  std::uint32_t min_lane0_low, min_lane0_high;
  std::uint32_t min_lane1_low, min_lane1_high;
  std::uint32_t min_lane3_low, min_lane3_high;
  std::uint32_t max_lane0_low, max_lane0_high;
  std::uint32_t max_lane1_low, max_lane1_high;
  std::uint32_t max_lane3_low, max_lane3_high;

  SplitU64(DoubleBits(2.0), &add_lane0_low, &add_lane0_high);
  SplitU64(DoubleBits(1.0), &add_lane1_low, &add_lane1_high);
  SplitU64(DoubleBits(3.0), &add_lane3_low, &add_lane3_high);
  SplitU64(DoubleBits(0.75), &mul_lane0_low, &mul_lane0_high);
  SplitU64(DoubleBits(-6.0), &mul_lane1_low, &mul_lane1_high);
  SplitU64(DoubleBits(-4.0), &mul_lane3_low, &mul_lane3_high);
  SplitU64(DoubleBits(0.5), &min_lane0_low, &min_lane0_high);
  SplitU64(DoubleBits(-2.0), &min_lane1_low, &min_lane1_high);
  SplitU64(DoubleBits(-1.0), &min_lane3_low, &min_lane3_high);
  SplitU64(DoubleBits(1.5), &max_lane0_low, &max_lane0_high);
  SplitU64(DoubleBits(3.0), &max_lane1_low, &max_lane1_high);
  SplitU64(DoubleBits(4.0), &max_lane3_low, &max_lane3_high);

  return state.vgprs[109][0] == add_lane0_low &&
         state.vgprs[109][1] == add_lane1_low &&
         state.vgprs[109][2] == 0xb9b9b9b9u &&
         state.vgprs[109][3] == add_lane3_low &&
         state.vgprs[110][0] == add_lane0_high &&
         state.vgprs[110][1] == add_lane1_high &&
         state.vgprs[110][2] == 0xc9c9c9c9u &&
         state.vgprs[110][3] == add_lane3_high &&
         state.vgprs[111][0] == mul_lane0_low &&
         state.vgprs[111][1] == mul_lane1_low &&
         state.vgprs[111][2] == 0xb1b1b1b1u &&
         state.vgprs[111][3] == mul_lane3_low &&
         state.vgprs[112][0] == mul_lane0_high &&
         state.vgprs[112][1] == mul_lane1_high &&
         state.vgprs[112][2] == 0xc1c1c1c1u &&
         state.vgprs[112][3] == mul_lane3_high &&
         state.vgprs[113][0] == min_lane0_low &&
         state.vgprs[113][1] == min_lane1_low &&
         state.vgprs[113][2] == 0xb3b3b3b3u &&
         state.vgprs[113][3] == min_lane3_low &&
         state.vgprs[114][0] == min_lane0_high &&
         state.vgprs[114][1] == min_lane1_high &&
         state.vgprs[114][2] == 0xc3c3c3c3u &&
         state.vgprs[114][3] == min_lane3_high &&
         state.vgprs[115][0] == max_lane0_low &&
         state.vgprs[115][1] == max_lane1_low &&
         state.vgprs[115][2] == 0xb5b5b5b5u &&
         state.vgprs[115][3] == max_lane3_low &&
         state.vgprs[116][0] == max_lane0_high &&
         state.vgprs[116][1] == max_lane1_high &&
         state.vgprs[116][2] == 0xc5c5c5c5u &&
         state.vgprs[116][3] == max_lane3_high &&
         state.exec_mask == 0xbu && state.halted &&
         !state.waiting_on_barrier && state.pc == 4u;
}

bool ExpectI24VectorBinarySeedState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.vgprs[117][0] == 0xfffeffefu &&
         state.vgprs[117][1] == 0xff000002u &&
         state.vgprs[117][2] == 0xd7d7d7d7u &&
         state.vgprs[117][3] == 0xffffffffu &&
         state.vgprs[118][0] == 0x00100000u &&
         state.vgprs[118][1] == 0xfffffffau &&
         state.vgprs[118][2] == 0xd8d8d8d8u &&
         state.vgprs[118][3] == 0xff000001u &&
         state.vgprs[119][0] == 0x00000000u &&
         state.vgprs[119][1] == 0xffffffffu &&
         state.vgprs[119][2] == 0xd9d9d9d9u &&
         state.vgprs[119][3] == 0x00003fffu &&
         state.vgprs[120][0] == 0x00100000u &&
         state.vgprs[120][1] == 0x02fffffau &&
         state.vgprs[120][2] == 0xdadadadau &&
         state.vgprs[120][3] == 0xff000001u &&
         state.vgprs[121][0] == 0x00000000u &&
         state.vgprs[121][1] == 0x00000000u &&
         state.vgprs[121][2] == 0xdbdbdbdbu &&
         state.vgprs[121][3] == 0x00003fffu &&
         state.exec_mask == 0xbu && state.halted &&
         !state.waiting_on_barrier && state.pc == 5u;
}

bool ExpectWideShiftSeedState(
    const mirage::sim::isa::WaveExecutionState& state) {
  std::uint32_t lane0_low, lane0_high;
  std::uint32_t lane1_low, lane1_high;
  std::uint32_t lane3_low, lane3_high;

  SplitU64(0x30ull, &lane0_low, &lane0_high);
  SplitU64(0x100ull, &lane1_low, &lane1_high);
  SplitU64(0x0000010000000000ull, &lane3_low, &lane3_high);

  return state.vgprs[124][0] == lane0_low &&
         state.vgprs[124][1] == lane1_low &&
         state.vgprs[124][2] == 0xe4e4e4e4u &&
         state.vgprs[124][3] == lane3_low &&
         state.vgprs[125][0] == lane0_high &&
         state.vgprs[125][1] == lane1_high &&
         state.vgprs[125][2] == 0xf5f5f5f5u &&
         state.vgprs[125][3] == lane3_high &&
         state.exec_mask == 0xbu && state.halted &&
         !state.waiting_on_barrier && state.pc == 1u;
}

constexpr std::uint64_t kGlobalAtomicExecMask = 0x80000005ull;
constexpr std::uint16_t kGlobalAtomicBaseSgpr = 24;
constexpr std::uint16_t kGlobalAtomicAddressVgpr = 10;
constexpr std::uint64_t kGlobalAtomicBaseAddress = 0xc000u;

std::int32_t BitsToI32(std::uint32_t value) {
  std::int32_t result = 0;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

std::uint32_t I32Bits(std::int32_t value) {
  std::uint32_t result = 0;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

std::int64_t BitsToI64(std::uint64_t value) {
  std::int64_t result = 0;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

std::uint64_t I64Bits(std::int64_t value) {
  std::uint64_t result = 0;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

bool StoreU64(mirage::sim::isa::LinearExecutionMemory* memory,
              std::uint64_t address,
              std::uint64_t value) {
  return memory->StoreU32(address, static_cast<std::uint32_t>(value)) &&
         memory->StoreU32(address + 4u,
                          static_cast<std::uint32_t>(value >> 32));
}

bool LoadU64(mirage::sim::isa::LinearExecutionMemory* memory,
             std::uint64_t address,
             std::uint64_t* value) {
  if (value == nullptr) {
    return false;
  }
  std::uint32_t low = 0;
  std::uint32_t high = 0;
  if (!memory->LoadU32(address, &low) || !memory->LoadU32(address + 4u, &high)) {
    return false;
  }
  *value = static_cast<std::uint64_t>(low) |
           (static_cast<std::uint64_t>(high) << 32);
  return true;
}

struct GlobalAtomicU32Case {
  const char* opcode;
  std::uint32_t op;
  mirage::sim::isa::Gfx1201CompiledOpcode compiled;
  std::uint16_t dst;
  std::uint16_t data;
  std::uint32_t offset;
  bool is_cmpswap;
};

constexpr std::array<GlobalAtomicU32Case, 15> kGlobalAtomicU32Cases{{
    {"GLOBAL_ATOMIC_SWAP_B32", 51u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicSwapB32, 30u, 50u,
     0x000u, false},
    {"GLOBAL_ATOMIC_CMPSWAP_B32", 52u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicCmpswapB32, 31u,
     52u, 0x100u, true},
    {"GLOBAL_ATOMIC_ADD_U32", 53u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicAddU32, 32u, 54u,
     0x200u, false},
    {"GLOBAL_ATOMIC_SUB_U32", 54u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicSubU32, 33u, 56u,
     0x300u, false},
    {"GLOBAL_ATOMIC_SUB_CLAMP_U32", 55u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicSubClampU32, 34u,
     58u, 0x400u, false},
    {"GLOBAL_ATOMIC_MIN_I32", 56u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicMinI32, 35u, 60u,
     0x500u, false},
    {"GLOBAL_ATOMIC_MIN_U32", 57u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicMinU32, 36u, 62u,
     0x600u, false},
    {"GLOBAL_ATOMIC_MAX_I32", 58u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicMaxI32, 37u, 64u,
     0x700u, false},
    {"GLOBAL_ATOMIC_MAX_U32", 59u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicMaxU32, 38u, 66u,
     0x800u, false},
    {"GLOBAL_ATOMIC_AND_B32", 60u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicAndB32, 39u, 68u,
     0x900u, false},
    {"GLOBAL_ATOMIC_OR_B32", 61u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicOrB32, 40u, 70u,
     0xa00u, false},
    {"GLOBAL_ATOMIC_XOR_B32", 62u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicXorB32, 41u, 72u,
     0xb00u, false},
    {"GLOBAL_ATOMIC_INC_U32", 63u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicIncU32, 42u, 74u,
     0xc00u, false},
    {"GLOBAL_ATOMIC_DEC_U32", 64u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicDecU32, 43u, 76u,
     0xd00u, false},
    {"GLOBAL_ATOMIC_COND_SUB_U32", 80u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicCondSubU32, 44u,
     78u, 0xe00u, false},
}};

constexpr std::uint16_t kGlobalAtomicU64BaseSgpr = 26;
constexpr std::uint16_t kGlobalAtomicU64AddressVgpr = 12;
constexpr std::uint64_t kGlobalAtomicU64BaseAddress = 0xd000u;

struct GlobalAtomicU64Case {
  const char* opcode;
  std::uint32_t op;
  mirage::sim::isa::Gfx1201CompiledOpcode compiled;
  std::uint16_t dst;
  std::uint16_t data;
  std::uint32_t offset;
  bool is_cmpswap;
};

constexpr std::array<GlobalAtomicU64Case, 14> kGlobalAtomicU64Cases{{
    {"GLOBAL_ATOMIC_SWAP_B64", 65u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicSwapB64, 20u, 60u,
     0x000u, false},
    {"GLOBAL_ATOMIC_CMPSWAP_B64", 66u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicCmpswapB64, 22u,
     64u, 0x100u, true},
    {"GLOBAL_ATOMIC_ADD_U64", 67u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicAddU64, 24u, 68u,
     0x200u, false},
    {"GLOBAL_ATOMIC_SUB_U64", 68u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicSubU64, 26u, 70u,
     0x300u, false},
    {"GLOBAL_ATOMIC_MIN_I64", 69u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicMinI64, 28u, 72u,
     0x400u, false},
    {"GLOBAL_ATOMIC_MIN_U64", 70u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicMinU64, 30u, 74u,
     0x500u, false},
    {"GLOBAL_ATOMIC_MAX_I64", 71u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicMaxI64, 32u, 76u,
     0x600u, false},
    {"GLOBAL_ATOMIC_MAX_U64", 72u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicMaxU64, 34u, 78u,
     0x700u, false},
    {"GLOBAL_ATOMIC_AND_B64", 73u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicAndB64, 36u, 80u,
     0x800u, false},
    {"GLOBAL_ATOMIC_OR_B64", 74u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicOrB64, 38u, 82u,
     0x900u, false},
    {"GLOBAL_ATOMIC_XOR_B64", 75u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicXorB64, 40u, 84u,
     0xa00u, false},
    {"GLOBAL_ATOMIC_INC_U64", 76u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicIncU64, 42u, 86u,
     0xb00u, false},
    {"GLOBAL_ATOMIC_DEC_U64", 77u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicDecU64, 44u, 88u,
     0xc00u, false},
    {"GLOBAL_ATOMIC_ORDERED_ADD_B64", 115u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicOrderedAddB64, 46u,
     90u, 0xd00u, false},
}};

constexpr std::uint16_t kGlobalAtomicF32BaseSgpr = 28;
constexpr std::uint16_t kGlobalAtomicF32AddressVgpr = 14;
constexpr std::uint64_t kGlobalAtomicF32BaseAddress = 0xe000u;

struct GlobalAtomicF32Case {
  const char* opcode;
  std::uint32_t op;
  mirage::sim::isa::Gfx1201CompiledOpcode compiled;
  std::uint16_t dst;
  std::uint16_t data;
  std::uint32_t offset;
};

constexpr std::array<GlobalAtomicF32Case, 3> kGlobalAtomicF32Cases{{
    {"GLOBAL_ATOMIC_ADD_F32", 86u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicAddF32, 90u, 96u,
     0x000u},
    {"GLOBAL_ATOMIC_MIN_NUM_F32", 81u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicMinNumF32, 91u,
     97u, 0x100u},
    {"GLOBAL_ATOMIC_MAX_NUM_F32", 82u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicMaxNumF32, 92u,
     98u, 0x200u},
}};

constexpr std::uint16_t kGlobalAtomicPackedBaseSgpr = 30;
constexpr std::uint16_t kGlobalAtomicPackedAddressVgpr = 16;
constexpr std::uint64_t kGlobalAtomicPackedBaseAddress = 0xf000u;

struct GlobalAtomicPackedCase {
  const char* opcode;
  std::uint32_t op;
  mirage::sim::isa::Gfx1201CompiledOpcode compiled;
  std::uint16_t dst;
  std::uint16_t data;
  std::uint32_t offset;
  bool use_bf16;
};

constexpr std::array<GlobalAtomicPackedCase, 2> kGlobalAtomicPackedCases{{
    {"GLOBAL_ATOMIC_PK_ADD_F16", 89u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicPkAddF16, 100u,
     104u, 0x000u, false},
    {"GLOBAL_ATOMIC_PK_ADD_BF16", 90u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kGlobalAtomicPkAddBf16, 101u,
     105u, 0x100u, true},
}};

bool IsGlobalAtomicLaneActive(std::size_t lane) {
  return (kGlobalAtomicExecMask & (1ull << lane)) != 0u;
}

std::uint32_t GlobalAtomicSentinel(std::size_t case_index, std::size_t lane) {
  return 0xea000000u + static_cast<std::uint32_t>(case_index << 8) +
         static_cast<std::uint32_t>(lane);
}

std::uint32_t InitialGlobalAtomicOldValue(std::size_t case_index,
                                          std::size_t lane) {
  if (!IsGlobalAtomicLaneActive(lane)) {
    return 0xf1000000u + static_cast<std::uint32_t>(case_index << 8) +
           static_cast<std::uint32_t>(lane);
  }

  switch (case_index) {
    case 0:
      return lane == 0u ? 0x10000010u
                        : (lane == 2u ? 0x10000020u : 0x10000030u);
    case 1:
      return lane == 0u ? 0x00000010u
                        : (lane == 2u ? 0x00000020u : 0x00000030u);
    case 2:
      return lane == 0u ? 10u : (lane == 2u ? 100u : 0xfffffff0u);
    case 3:
      return lane == 0u ? 20u : (lane == 2u ? 5u : 1u);
    case 4:
      return lane == 0u ? 4u : (lane == 2u ? 10u : 1u);
    case 5:
      return lane == 0u ? I32Bits(-5) : (lane == 2u ? 9u : I32Bits(-1));
    case 6:
      return lane == 0u ? 5u : (lane == 2u ? 9u : 0xfffffff0u);
    case 7:
      return lane == 0u ? I32Bits(-5) : (lane == 2u ? 9u : I32Bits(-1));
    case 8:
      return lane == 0u ? 5u : (lane == 2u ? 9u : 0xfffffff0u);
    case 9:
      return lane == 0u ? 0x0f0f00f0u
                        : (lane == 2u ? 0xff00ff00u : 0xaaaaaaaau);
    case 10:
      return lane == 0u ? 0x000000f0u
                        : (lane == 2u ? 0x00ff0000u : 0x0f000f00u);
    case 11:
      return lane == 0u ? 0xaaaa5555u
                        : (lane == 2u ? 0x12345678u : 0xffffffffu);
    case 12:
      return lane == 0u ? 4u : (lane == 2u ? 3u : 8u);
    case 13:
      return lane == 0u ? 0u : (lane == 2u ? 5u : 4u);
    case 14:
      return lane == 0u ? 7u : (lane == 2u ? 4u : 5u);
    default:
      return 0u;
  }
}

std::uint32_t InitialGlobalAtomicDataValue(std::size_t case_index,
                                           std::size_t lane) {
  if (!IsGlobalAtomicLaneActive(lane)) {
    return 0x51000000u + static_cast<std::uint32_t>(case_index << 8) +
           static_cast<std::uint32_t>(lane);
  }

  switch (case_index) {
    case 0:
      return lane == 0u ? 0x90000010u
                        : (lane == 2u ? 0x90000020u : 0x90000030u);
    case 1:
      return lane == 0u ? 0x00000010u
                        : (lane == 2u ? 0x00000099u : 0x00000030u);
    case 2:
      return lane == 0u ? 5u : (lane == 2u ? 7u : 0x20u);
    case 3:
      return lane == 0u ? 3u : (lane == 2u ? 7u : 2u);
    case 4:
      return lane == 0u ? 7u : (lane == 2u ? 6u : 1u);
    case 5:
      return lane == 0u ? I32Bits(-2) : (lane == 2u ? I32Bits(-4) : 7u);
    case 6:
      return lane == 0u ? 2u : (lane == 2u ? 10u : 0x10u);
    case 7:
      return lane == 0u ? I32Bits(-2) : (lane == 2u ? I32Bits(-4) : 7u);
    case 8:
      return lane == 0u ? 2u : (lane == 2u ? 10u : 0x10u);
    case 9:
      return lane == 0u ? 0x00ff0ff0u
                        : (lane == 2u ? 0x0f0f0f0fu : 0x00ffff00u);
    case 10:
      return lane == 0u ? 0x00000f00u
                        : (lane == 2u ? 0x000000ffu : 0xf000000fu);
    case 11:
      return lane == 0u ? 0xffff0000u
                        : (lane == 2u ? 0x0f0f0f0fu : 0x12345678u);
    case 12:
      return 4u;
    case 13:
      return 4u;
    case 14:
      return 5u;
    default:
      return 0u;
  }
}

std::uint32_t InitialGlobalAtomicReplacementValue(std::size_t case_index,
                                                  std::size_t lane) {
  if (case_index != 1u) {
    return 0u;
  }
  if (!IsGlobalAtomicLaneActive(lane)) {
    return 0x52000000u + static_cast<std::uint32_t>(lane);
  }
  return lane == 0u ? 0xaaaa0001u
                    : (lane == 2u ? 0xbbbb0002u : 0xcccc0003u);
}

std::uint32_t ExpectedGlobalAtomicNewValue(std::size_t case_index,
                                           std::uint32_t old_value,
                                           std::uint32_t data_value,
                                           std::uint32_t replacement_value) {
  switch (case_index) {
    case 0:
      return data_value;
    case 1:
      return old_value == data_value ? replacement_value : old_value;
    case 2:
      return old_value + data_value;
    case 3:
      return old_value - data_value;
    case 4:
      return data_value > old_value ? data_value - old_value : 0u;
    case 5:
      return I32Bits(std::min(BitsToI32(old_value), BitsToI32(data_value)));
    case 6:
      return std::min(old_value, data_value);
    case 7:
      return I32Bits(std::max(BitsToI32(old_value), BitsToI32(data_value)));
    case 8:
      return std::max(old_value, data_value);
    case 9:
      return old_value & data_value;
    case 10:
      return old_value | data_value;
    case 11:
      return old_value ^ data_value;
    case 12:
      return old_value >= data_value ? 0u : old_value + 1u;
    case 13:
      return (old_value == 0u || old_value > data_value) ? data_value
                                                          : old_value - 1u;
    case 14:
      return old_value >= data_value ? old_value - data_value : old_value;
    default:
      return old_value;
  }
}

bool RunGlobalAtomicU32BatchTest(
    const mirage::sim::isa::Gfx1201BinaryDecoder& decoder,
    const mirage::sim::isa::Gfx1201Interpreter& interpreter,
    std::string* error_message) {
  using namespace mirage::sim::isa;

  std::vector<std::uint32_t> atomic_program_words;
  atomic_program_words.reserve(kGlobalAtomicU32Cases.size() * 2u + 1u);
  for (const GlobalAtomicU32Case& atomic_case : kGlobalAtomicU32Cases) {
    const auto words =
        MakeGlobal(atomic_case.op, atomic_case.dst, kGlobalAtomicAddressVgpr,
                   atomic_case.data, kGlobalAtomicBaseSgpr, atomic_case.offset);
    atomic_program_words.push_back(words[0]);
    atomic_program_words.push_back(words[1]);
  }
  atomic_program_words.push_back(MakeSopp(48u));

  std::vector<DecodedInstruction> atomic_program;
  if (!Expect(decoder.DecodeProgram(atomic_program_words, &atomic_program,
                                    error_message),
              "expected GLOBAL atomic program decode success") ||
      !Expect(atomic_program.size() == kGlobalAtomicU32Cases.size() + 1u,
              "expected decoded GLOBAL atomic instruction count")) {
    return false;
  }

  for (std::size_t i = 0; i < kGlobalAtomicU32Cases.size(); ++i) {
    if (!Expect(atomic_program[i].opcode == kGlobalAtomicU32Cases[i].opcode,
                "expected decoded GLOBAL atomic opcode order")) {
      return false;
    }
  }
  if (!Expect(atomic_program.back().opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after GLOBAL atomic batch")) {
    return false;
  }

  auto initialize_atomic_state = [](WaveExecutionState* state) {
    state->exec_mask = kGlobalAtomicExecMask;
    state->sgprs[kGlobalAtomicBaseSgpr] =
        static_cast<std::uint32_t>(kGlobalAtomicBaseAddress);
    state->sgprs[kGlobalAtomicBaseSgpr + 1u] = 0u;

    for (std::size_t lane = 0; lane < 32u; ++lane) {
      state->vgprs[kGlobalAtomicAddressVgpr][lane] =
          static_cast<std::uint32_t>(lane * 4u);
      for (std::size_t case_index = 0; case_index < kGlobalAtomicU32Cases.size();
           ++case_index) {
        const GlobalAtomicU32Case& atomic_case = kGlobalAtomicU32Cases[case_index];
        state->vgprs[atomic_case.dst][lane] =
            GlobalAtomicSentinel(case_index, lane);
        state->vgprs[atomic_case.data][lane] =
            InitialGlobalAtomicDataValue(case_index, lane);
        if (atomic_case.is_cmpswap) {
          state->vgprs[atomic_case.data + 1u][lane] =
              InitialGlobalAtomicReplacementValue(case_index, lane);
        }
      }
    }
  };

  auto expect_atomic_state = [](const WaveExecutionState& state) {
    if (!(state.lane_count == 32u && state.exec_mask == kGlobalAtomicExecMask &&
          state.sgprs[kGlobalAtomicBaseSgpr] ==
              static_cast<std::uint32_t>(kGlobalAtomicBaseAddress) &&
          state.sgprs[kGlobalAtomicBaseSgpr + 1u] == 0u && state.halted &&
          !state.waiting_on_barrier &&
          state.pc == kGlobalAtomicU32Cases.size())) {
      return false;
    }

    for (std::size_t lane = 0; lane < 32u; ++lane) {
      if (state.vgprs[kGlobalAtomicAddressVgpr][lane] !=
          static_cast<std::uint32_t>(lane * 4u)) {
        return false;
      }
      for (std::size_t case_index = 0; case_index < kGlobalAtomicU32Cases.size();
           ++case_index) {
        const GlobalAtomicU32Case& atomic_case = kGlobalAtomicU32Cases[case_index];
        const std::uint32_t expected_dst =
            IsGlobalAtomicLaneActive(lane)
                ? InitialGlobalAtomicOldValue(case_index, lane)
                : GlobalAtomicSentinel(case_index, lane);
        if (state.vgprs[atomic_case.dst][lane] != expected_dst ||
            state.vgprs[atomic_case.data][lane] !=
                InitialGlobalAtomicDataValue(case_index, lane)) {
          return false;
        }
        if (atomic_case.is_cmpswap &&
            state.vgprs[atomic_case.data + 1u][lane] !=
                InitialGlobalAtomicReplacementValue(case_index, lane)) {
          return false;
        }
      }
    }
    return true;
  };

  auto initialize_atomic_memory = [](LinearExecutionMemory* memory) {
    for (std::size_t case_index = 0; case_index < kGlobalAtomicU32Cases.size();
         ++case_index) {
      const GlobalAtomicU32Case& atomic_case = kGlobalAtomicU32Cases[case_index];
      for (std::size_t lane = 0; lane < 32u; ++lane) {
        const std::uint64_t address =
            kGlobalAtomicBaseAddress + atomic_case.offset + lane * 4u;
        if (!memory->StoreU32(address,
                              InitialGlobalAtomicOldValue(case_index, lane))) {
          return false;
        }
      }
    }
    return true;
  };

  auto expect_atomic_memory = [](LinearExecutionMemory* memory) {
    for (std::size_t case_index = 0; case_index < kGlobalAtomicU32Cases.size();
         ++case_index) {
      const GlobalAtomicU32Case& atomic_case = kGlobalAtomicU32Cases[case_index];
      for (std::size_t lane = 0; lane < 32u; ++lane) {
        const std::uint64_t address =
            kGlobalAtomicBaseAddress + atomic_case.offset + lane * 4u;
        std::uint32_t value = 0;
        if (!memory->LoadU32(address, &value)) {
          return false;
        }
        const std::uint32_t old_value =
            InitialGlobalAtomicOldValue(case_index, lane);
        const std::uint32_t expected_value =
            IsGlobalAtomicLaneActive(lane)
                ? ExpectedGlobalAtomicNewValue(
                      case_index, old_value,
                      InitialGlobalAtomicDataValue(case_index, lane),
                      InitialGlobalAtomicReplacementValue(case_index, lane))
                : old_value;
        if (value != expected_value) {
          return false;
        }
      }
    }
    return true;
  };

  LinearExecutionMemory decoded_atomic_memory(0x1000u,
                                              kGlobalAtomicBaseAddress);
  if (!Expect(initialize_atomic_memory(&decoded_atomic_memory),
              "expected GLOBAL atomic decoded memory initialization")) {
    return false;
  }
  WaveExecutionState decoded_atomic_state;
  initialize_atomic_state(&decoded_atomic_state);
  if (!Expect(interpreter.ExecuteProgram(atomic_program, &decoded_atomic_state,
                                         &decoded_atomic_memory, error_message),
              "expected decoded GLOBAL atomic execution success") ||
      !Expect(expect_atomic_state(decoded_atomic_state),
              "expected decoded GLOBAL atomic state") ||
      !Expect(expect_atomic_memory(&decoded_atomic_memory),
              "expected decoded GLOBAL atomic memory state")) {
    return false;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_atomic_program;
  if (!Expect(interpreter.CompileProgram(atomic_program, &compiled_atomic_program,
                                         error_message),
              "expected compiled GLOBAL atomic program success") ||
      !Expect(compiled_atomic_program.size() == kGlobalAtomicU32Cases.size() + 1u,
              "expected compiled GLOBAL atomic instruction count")) {
    return false;
  }

  for (std::size_t i = 0; i < kGlobalAtomicU32Cases.size(); ++i) {
    if (!Expect(compiled_atomic_program[i].opcode ==
                    kGlobalAtomicU32Cases[i].compiled,
                "expected compiled GLOBAL atomic opcode order")) {
      return false;
    }
  }
  if (!Expect(compiled_atomic_program.back().opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after GLOBAL atomic batch")) {
    return false;
  }

  LinearExecutionMemory compiled_atomic_memory(0x1000u,
                                               kGlobalAtomicBaseAddress);
  if (!Expect(initialize_atomic_memory(&compiled_atomic_memory),
              "expected GLOBAL atomic compiled memory initialization")) {
    return false;
  }
  WaveExecutionState compiled_atomic_state;
  initialize_atomic_state(&compiled_atomic_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_atomic_program,
                                         &compiled_atomic_state,
                                         &compiled_atomic_memory,
                                         error_message),
              "expected compiled GLOBAL atomic execution success") ||
      !Expect(expect_atomic_state(compiled_atomic_state),
              "expected compiled GLOBAL atomic state") ||
      !Expect(expect_atomic_memory(&compiled_atomic_memory),
              "expected compiled GLOBAL atomic memory state")) {
    return false;
  }

  return true;
}

std::uint64_t GlobalAtomicU64Sentinel(std::size_t case_index, std::size_t lane) {
  return 0xea00000000000000ull +
         (static_cast<std::uint64_t>(case_index) << 8) +
         static_cast<std::uint64_t>(lane);
}

std::uint64_t InitialGlobalAtomicU64OldValue(std::size_t case_index,
                                             std::size_t lane) {
  if (!IsGlobalAtomicLaneActive(lane)) {
    return 0xf100000000000000ull +
           (static_cast<std::uint64_t>(case_index) << 8) +
           static_cast<std::uint64_t>(lane);
  }

  switch (case_index) {
    case 0:
      return lane == 0u ? 0x1000000000000010ull
                        : (lane == 2u ? 0x2000000000000020ull
                                       : 0x3000000000000030ull);
    case 1:
      return lane == 0u ? 0x0000000000000010ull
                        : (lane == 2u ? 0x0000000000000020ull
                                       : 0x0000000000000030ull);
    case 2:
      return lane == 0u ? 10ull
                        : (lane == 2u ? 100ull : 0xfffffffffffffff0ull);
    case 3:
      return lane == 0u ? 20ull : (lane == 2u ? 5ull : 1ull);
    case 4:
      return lane == 0u ? I64Bits(-5ll)
                        : (lane == 2u ? 9ull : I64Bits(-1ll));
    case 5:
      return lane == 0u ? 5ull
                        : (lane == 2u ? 9ull : 0xfffffffffffffff0ull);
    case 6:
      return lane == 0u ? I64Bits(-5ll)
                        : (lane == 2u ? 9ull : I64Bits(-1ll));
    case 7:
      return lane == 0u ? 5ull
                        : (lane == 2u ? 9ull : 0xfffffffffffffff0ull);
    case 8:
      return lane == 0u ? 0x0f0f00f00f0f00f0ull
                        : (lane == 2u ? 0xff00ff00ff00ff00ull
                                       : 0xaaaaaaaa55555555ull);
    case 9:
      return lane == 0u ? 0x000000f0000000f0ull
                        : (lane == 2u ? 0x00ff000000ff0000ull
                                       : 0x0f000f000f000f00ull);
    case 10:
      return lane == 0u ? 0xaaaa5555aaaa5555ull
                        : (lane == 2u ? 0x123456789abcdef0ull
                                       : 0xffffffffffffffffull);
    case 11:
      return lane == 0u ? 4ull : (lane == 2u ? 3ull : 8ull);
    case 12:
      return lane == 0u ? 0ull : (lane == 2u ? 5ull : 4ull);
    case 13:
      return lane == 0u ? 11ull
                        : (lane == 2u ? 0x100ull : 0xfffffffffffffff0ull);
    default:
      return 0ull;
  }
}

std::uint64_t InitialGlobalAtomicU64DataValue(std::size_t case_index,
                                              std::size_t lane) {
  if (!IsGlobalAtomicLaneActive(lane)) {
    return 0x5100000000000000ull +
           (static_cast<std::uint64_t>(case_index) << 8) +
           static_cast<std::uint64_t>(lane);
  }

  switch (case_index) {
    case 0:
      return lane == 0u ? 0x9000000000000010ull
                        : (lane == 2u ? 0x9000000000000020ull
                                       : 0x9000000000000030ull);
    case 1:
      return lane == 0u ? 0x0000000000000010ull
                        : (lane == 2u ? 0x0000000000000099ull
                                       : 0x0000000000000030ull);
    case 2:
      return lane == 0u ? 5ull : (lane == 2u ? 7ull : 0x20ull);
    case 3:
      return lane == 0u ? 3ull : (lane == 2u ? 7ull : 2ull);
    case 4:
      return lane == 0u ? I64Bits(-2ll)
                        : (lane == 2u ? I64Bits(-4ll) : 7ull);
    case 5:
      return lane == 0u ? 2ull : (lane == 2u ? 10ull : 0x10ull);
    case 6:
      return lane == 0u ? I64Bits(-2ll)
                        : (lane == 2u ? I64Bits(-4ll) : 7ull);
    case 7:
      return lane == 0u ? 2ull : (lane == 2u ? 10ull : 0x10ull);
    case 8:
      return lane == 0u ? 0x00ff0ff000ff0ff0ull
                        : (lane == 2u ? 0x0f0f0f0f0f0f0f0full
                                       : 0x00ffff0000ffff00ull);
    case 9:
      return lane == 0u ? 0x00000f0000000f00ull
                        : (lane == 2u ? 0x000000ff000000ffull
                                       : 0xf000000ff000000full);
    case 10:
      return lane == 0u ? 0xffff0000ffff0000ull
                        : (lane == 2u ? 0x0f0f0f0f0f0f0f0full
                                       : 0x123456789abcdef0ull);
    case 11:
      return 4ull;
    case 12:
      return 4ull;
    case 13:
      return lane == 0u ? 5ull : (lane == 2u ? 7ull : 0x20ull);
    default:
      return 0ull;
  }
}

std::uint64_t InitialGlobalAtomicU64ReplacementValue(std::size_t case_index,
                                                     std::size_t lane) {
  if (case_index != 1u) {
    return 0ull;
  }
  if (!IsGlobalAtomicLaneActive(lane)) {
    return 0x5200000000000000ull + static_cast<std::uint64_t>(lane);
  }
  return lane == 0u ? 0xaaaa000000000001ull
                    : (lane == 2u ? 0xbbbb000000000002ull
                                   : 0xcccc000000000003ull);
}

std::uint64_t ExpectedGlobalAtomicU64NewValue(std::size_t case_index,
                                              std::uint64_t old_value,
                                              std::uint64_t data_value,
                                              std::uint64_t replacement_value) {
  switch (case_index) {
    case 0:
      return data_value;
    case 1:
      return old_value == data_value ? replacement_value : old_value;
    case 2:
      return old_value + data_value;
    case 3:
      return old_value - data_value;
    case 4:
      return I64Bits(std::min(BitsToI64(old_value), BitsToI64(data_value)));
    case 5:
      return std::min(old_value, data_value);
    case 6:
      return I64Bits(std::max(BitsToI64(old_value), BitsToI64(data_value)));
    case 7:
      return std::max(old_value, data_value);
    case 8:
      return old_value & data_value;
    case 9:
      return old_value | data_value;
    case 10:
      return old_value ^ data_value;
    case 11:
      return old_value >= data_value ? 0ull : old_value + 1ull;
    case 12:
      return (old_value == 0ull || old_value > data_value) ? data_value
                                                            : old_value - 1ull;
    case 13:
      return old_value + data_value;
    default:
      return old_value;
  }
}

bool RunGlobalAtomicU64BatchTest(
    const mirage::sim::isa::Gfx1201BinaryDecoder& decoder,
    const mirage::sim::isa::Gfx1201Interpreter& interpreter,
    std::string* error_message) {
  using namespace mirage::sim::isa;

  std::vector<std::uint32_t> atomic_program_words;
  atomic_program_words.reserve(kGlobalAtomicU64Cases.size() * 2u + 1u);
  for (const GlobalAtomicU64Case& atomic_case : kGlobalAtomicU64Cases) {
    const auto words =
        MakeGlobal(atomic_case.op, atomic_case.dst, kGlobalAtomicU64AddressVgpr,
                   atomic_case.data, kGlobalAtomicU64BaseSgpr,
                   atomic_case.offset);
    atomic_program_words.push_back(words[0]);
    atomic_program_words.push_back(words[1]);
  }
  atomic_program_words.push_back(MakeSopp(48u));

  std::vector<DecodedInstruction> atomic_program;
  if (!Expect(decoder.DecodeProgram(atomic_program_words, &atomic_program,
                                    error_message),
              "expected GLOBAL atomic B64 program decode success") ||
      !Expect(atomic_program.size() == kGlobalAtomicU64Cases.size() + 1u,
              "expected decoded GLOBAL atomic B64 instruction count")) {
    return false;
  }

  for (std::size_t i = 0; i < kGlobalAtomicU64Cases.size(); ++i) {
    if (!Expect(atomic_program[i].opcode == kGlobalAtomicU64Cases[i].opcode,
                "expected decoded GLOBAL atomic B64 opcode order")) {
      return false;
    }
  }
  if (!Expect(atomic_program.back().opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after GLOBAL atomic B64 batch")) {
    return false;
  }

  auto initialize_atomic_state = [](WaveExecutionState* state) {
    state->exec_mask = kGlobalAtomicExecMask;
    state->sgprs[kGlobalAtomicU64BaseSgpr] =
        static_cast<std::uint32_t>(kGlobalAtomicU64BaseAddress);
    state->sgprs[kGlobalAtomicU64BaseSgpr + 1u] = 0u;

    for (std::size_t lane = 0; lane < 32u; ++lane) {
      state->vgprs[kGlobalAtomicU64AddressVgpr][lane] =
          static_cast<std::uint32_t>(lane * 8u);
      for (std::size_t case_index = 0; case_index < kGlobalAtomicU64Cases.size();
           ++case_index) {
        const GlobalAtomicU64Case& atomic_case = kGlobalAtomicU64Cases[case_index];
        const std::uint64_t sentinel = GlobalAtomicU64Sentinel(case_index, lane);
        state->vgprs[atomic_case.dst][lane] = static_cast<std::uint32_t>(sentinel);
        state->vgprs[atomic_case.dst + 1u][lane] =
            static_cast<std::uint32_t>(sentinel >> 32);

        const std::uint64_t data_value =
            InitialGlobalAtomicU64DataValue(case_index, lane);
        state->vgprs[atomic_case.data][lane] = static_cast<std::uint32_t>(data_value);
        state->vgprs[atomic_case.data + 1u][lane] =
            static_cast<std::uint32_t>(data_value >> 32);
        if (atomic_case.is_cmpswap) {
          const std::uint64_t replacement =
              InitialGlobalAtomicU64ReplacementValue(case_index, lane);
          state->vgprs[atomic_case.data + 2u][lane] =
              static_cast<std::uint32_t>(replacement);
          state->vgprs[atomic_case.data + 3u][lane] =
              static_cast<std::uint32_t>(replacement >> 32);
        }
      }
    }
  };

  auto expect_atomic_state = [](const WaveExecutionState& state) {
    if (!(state.lane_count == 32u && state.exec_mask == kGlobalAtomicExecMask &&
          state.sgprs[kGlobalAtomicU64BaseSgpr] ==
              static_cast<std::uint32_t>(kGlobalAtomicU64BaseAddress) &&
          state.sgprs[kGlobalAtomicU64BaseSgpr + 1u] == 0u && state.halted &&
          !state.waiting_on_barrier && state.pc == kGlobalAtomicU64Cases.size())) {
      return false;
    }

    for (std::size_t lane = 0; lane < 32u; ++lane) {
      if (state.vgprs[kGlobalAtomicU64AddressVgpr][lane] !=
          static_cast<std::uint32_t>(lane * 8u)) {
        return false;
      }
      for (std::size_t case_index = 0; case_index < kGlobalAtomicU64Cases.size();
           ++case_index) {
        const GlobalAtomicU64Case& atomic_case = kGlobalAtomicU64Cases[case_index];
        const std::uint64_t expected_dst =
            IsGlobalAtomicLaneActive(lane)
                ? InitialGlobalAtomicU64OldValue(case_index, lane)
                : GlobalAtomicU64Sentinel(case_index, lane);
        if (state.vgprs[atomic_case.dst][lane] !=
                static_cast<std::uint32_t>(expected_dst) ||
            state.vgprs[atomic_case.dst + 1u][lane] !=
                static_cast<std::uint32_t>(expected_dst >> 32)) {
          return false;
        }

        const std::uint64_t data_value =
            InitialGlobalAtomicU64DataValue(case_index, lane);
        if (state.vgprs[atomic_case.data][lane] !=
                static_cast<std::uint32_t>(data_value) ||
            state.vgprs[atomic_case.data + 1u][lane] !=
                static_cast<std::uint32_t>(data_value >> 32)) {
          return false;
        }

        if (atomic_case.is_cmpswap) {
          const std::uint64_t replacement =
              InitialGlobalAtomicU64ReplacementValue(case_index, lane);
          if (state.vgprs[atomic_case.data + 2u][lane] !=
                  static_cast<std::uint32_t>(replacement) ||
              state.vgprs[atomic_case.data + 3u][lane] !=
                  static_cast<std::uint32_t>(replacement >> 32)) {
            return false;
          }
        }
      }
    }
    return true;
  };

  auto initialize_atomic_memory = [](LinearExecutionMemory* memory) {
    for (std::size_t case_index = 0; case_index < kGlobalAtomicU64Cases.size();
         ++case_index) {
      const GlobalAtomicU64Case& atomic_case = kGlobalAtomicU64Cases[case_index];
      for (std::size_t lane = 0; lane < 32u; ++lane) {
        const std::uint64_t address =
            kGlobalAtomicU64BaseAddress + atomic_case.offset + lane * 8u;
        if (!StoreU64(memory, address,
                      InitialGlobalAtomicU64OldValue(case_index, lane))) {
          return false;
        }
      }
    }
    return true;
  };

  auto expect_atomic_memory = [](LinearExecutionMemory* memory) {
    for (std::size_t case_index = 0; case_index < kGlobalAtomicU64Cases.size();
         ++case_index) {
      const GlobalAtomicU64Case& atomic_case = kGlobalAtomicU64Cases[case_index];
      for (std::size_t lane = 0; lane < 32u; ++lane) {
        const std::uint64_t address =
            kGlobalAtomicU64BaseAddress + atomic_case.offset + lane * 8u;
        std::uint64_t value = 0;
        if (!LoadU64(memory, address, &value)) {
          return false;
        }
        const std::uint64_t old_value =
            InitialGlobalAtomicU64OldValue(case_index, lane);
        const std::uint64_t expected_value =
            IsGlobalAtomicLaneActive(lane)
                ? ExpectedGlobalAtomicU64NewValue(
                      case_index, old_value,
                      InitialGlobalAtomicU64DataValue(case_index, lane),
                      InitialGlobalAtomicU64ReplacementValue(case_index, lane))
                : old_value;
        if (value != expected_value) {
          return false;
        }
      }
    }
    return true;
  };

  LinearExecutionMemory decoded_atomic_memory(0x2000u, kGlobalAtomicU64BaseAddress);
  if (!Expect(initialize_atomic_memory(&decoded_atomic_memory),
              "expected GLOBAL atomic B64 decoded memory initialization")) {
    return false;
  }
  WaveExecutionState decoded_atomic_state;
  initialize_atomic_state(&decoded_atomic_state);
  if (!Expect(interpreter.ExecuteProgram(atomic_program, &decoded_atomic_state,
                                         &decoded_atomic_memory, error_message),
              "expected decoded GLOBAL atomic B64 execution success") ||
      !Expect(expect_atomic_state(decoded_atomic_state),
              "expected decoded GLOBAL atomic B64 state") ||
      !Expect(expect_atomic_memory(&decoded_atomic_memory),
              "expected decoded GLOBAL atomic B64 memory state")) {
    return false;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_atomic_program;
  if (!Expect(interpreter.CompileProgram(atomic_program, &compiled_atomic_program,
                                         error_message),
              "expected compiled GLOBAL atomic B64 program success") ||
      !Expect(compiled_atomic_program.size() == kGlobalAtomicU64Cases.size() + 1u,
              "expected compiled GLOBAL atomic B64 instruction count")) {
    return false;
  }

  for (std::size_t i = 0; i < kGlobalAtomicU64Cases.size(); ++i) {
    if (!Expect(compiled_atomic_program[i].opcode ==
                    kGlobalAtomicU64Cases[i].compiled,
                "expected compiled GLOBAL atomic B64 opcode order")) {
      return false;
    }
  }
  if (!Expect(compiled_atomic_program.back().opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after GLOBAL atomic B64 batch")) {
    return false;
  }

  LinearExecutionMemory compiled_atomic_memory(0x2000u,
                                               kGlobalAtomicU64BaseAddress);
  if (!Expect(initialize_atomic_memory(&compiled_atomic_memory),
              "expected GLOBAL atomic B64 compiled memory initialization")) {
    return false;
  }
  WaveExecutionState compiled_atomic_state;
  initialize_atomic_state(&compiled_atomic_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_atomic_program,
                                         &compiled_atomic_state,
                                         &compiled_atomic_memory,
                                         error_message),
              "expected compiled GLOBAL atomic B64 execution success") ||
      !Expect(expect_atomic_state(compiled_atomic_state),
              "expected compiled GLOBAL atomic B64 state") ||
      !Expect(expect_atomic_memory(&compiled_atomic_memory),
              "expected compiled GLOBAL atomic B64 memory state")) {
    return false;
  }

  return true;
}

std::uint32_t GlobalAtomicF32Sentinel(std::size_t case_index, std::size_t lane) {
  return 0xec000000u + static_cast<std::uint32_t>(case_index << 8) +
         static_cast<std::uint32_t>(lane);
}

std::uint32_t InitialGlobalAtomicF32OldValue(std::size_t case_index,
                                             std::size_t lane) {
  if (!IsGlobalAtomicLaneActive(lane)) {
    return 0xf3000000u + static_cast<std::uint32_t>(case_index << 8) +
           static_cast<std::uint32_t>(lane);
  }

  switch (case_index) {
    case 0:
      return lane == 0u ? FloatBits(1.5f)
                        : (lane == 2u ? FloatBits(-4.5f)
                                       : FloatBits(8.0f));
    case 1:
      return lane == 0u ? kQuietNaNF32Bits
                        : (lane == 2u ? FloatBits(4.0f)
                                       : FloatBits(-2.0f));
    case 2:
      return lane == 0u ? FloatBits(1.0f)
                        : (lane == 2u ? kQuietNaNF32Bits
                                       : FloatBits(-2.0f));
    default:
      return 0u;
  }
}

std::uint32_t InitialGlobalAtomicF32DataValue(std::size_t case_index,
                                              std::size_t lane) {
  if (!IsGlobalAtomicLaneActive(lane)) {
    return 0x53000000u + static_cast<std::uint32_t>(case_index << 8) +
           static_cast<std::uint32_t>(lane);
  }

  switch (case_index) {
    case 0:
      return lane == 0u ? FloatBits(2.25f)
                        : (lane == 2u ? FloatBits(1.0f)
                                       : FloatBits(-1.5f));
    case 1:
      return lane == 0u ? FloatBits(3.0f)
                        : (lane == 2u ? kQuietNaNF32Bits
                                       : FloatBits(5.0f));
    case 2:
      return lane == 0u ? FloatBits(3.0f)
                        : (lane == 2u ? FloatBits(7.0f)
                                       : kQuietNaNF32Bits);
    default:
      return 0u;
  }
}

std::uint32_t ExpectedGlobalAtomicF32NewValue(std::size_t case_index,
                                              std::uint32_t old_value,
                                              std::uint32_t data_value) {
  switch (case_index) {
    case 0:
      return FloatBits(BitsToFloat(old_value) + BitsToFloat(data_value));
    case 1:
      return FloatBits(std::fmin(BitsToFloat(old_value), BitsToFloat(data_value)));
    case 2:
      return FloatBits(std::fmax(BitsToFloat(old_value), BitsToFloat(data_value)));
    default:
      return old_value;
  }
}

bool RunGlobalAtomicF32BatchTest(
    const mirage::sim::isa::Gfx1201BinaryDecoder& decoder,
    const mirage::sim::isa::Gfx1201Interpreter& interpreter,
    std::string* error_message) {
  using namespace mirage::sim::isa;

  std::vector<std::uint32_t> atomic_program_words;
  atomic_program_words.reserve(kGlobalAtomicF32Cases.size() * 2u + 1u);
  for (const GlobalAtomicF32Case& atomic_case : kGlobalAtomicF32Cases) {
    const auto words =
        MakeGlobal(atomic_case.op, atomic_case.dst, kGlobalAtomicF32AddressVgpr,
                   atomic_case.data, kGlobalAtomicF32BaseSgpr,
                   atomic_case.offset);
    atomic_program_words.push_back(words[0]);
    atomic_program_words.push_back(words[1]);
  }
  atomic_program_words.push_back(MakeSopp(48u));

  std::vector<DecodedInstruction> atomic_program;
  if (!Expect(decoder.DecodeProgram(atomic_program_words, &atomic_program,
                                    error_message),
              "expected GLOBAL atomic F32 program decode success") ||
      !Expect(atomic_program.size() == kGlobalAtomicF32Cases.size() + 1u,
              "expected decoded GLOBAL atomic F32 instruction count")) {
    return false;
  }

  for (std::size_t i = 0; i < kGlobalAtomicF32Cases.size(); ++i) {
    if (!Expect(atomic_program[i].opcode == kGlobalAtomicF32Cases[i].opcode,
                "expected decoded GLOBAL atomic F32 opcode order")) {
      return false;
    }
  }
  if (!Expect(atomic_program.back().opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after GLOBAL atomic F32 batch")) {
    return false;
  }

  auto initialize_atomic_state = [](WaveExecutionState* state) {
    state->exec_mask = kGlobalAtomicExecMask;
    state->sgprs[kGlobalAtomicF32BaseSgpr] =
        static_cast<std::uint32_t>(kGlobalAtomicF32BaseAddress);
    state->sgprs[kGlobalAtomicF32BaseSgpr + 1u] = 0u;

    for (std::size_t lane = 0; lane < 32u; ++lane) {
      state->vgprs[kGlobalAtomicF32AddressVgpr][lane] =
          static_cast<std::uint32_t>(lane * 4u);
      for (std::size_t case_index = 0; case_index < kGlobalAtomicF32Cases.size();
           ++case_index) {
        const GlobalAtomicF32Case& atomic_case = kGlobalAtomicF32Cases[case_index];
        state->vgprs[atomic_case.dst][lane] =
            GlobalAtomicF32Sentinel(case_index, lane);
        state->vgprs[atomic_case.data][lane] =
            InitialGlobalAtomicF32DataValue(case_index, lane);
      }
    }
  };

  auto expect_atomic_state = [](const WaveExecutionState& state) {
    if (!(state.lane_count == 32u && state.exec_mask == kGlobalAtomicExecMask &&
          state.sgprs[kGlobalAtomicF32BaseSgpr] ==
              static_cast<std::uint32_t>(kGlobalAtomicF32BaseAddress) &&
          state.sgprs[kGlobalAtomicF32BaseSgpr + 1u] == 0u && state.halted &&
          !state.waiting_on_barrier && state.pc == kGlobalAtomicF32Cases.size())) {
      return false;
    }

    for (std::size_t lane = 0; lane < 32u; ++lane) {
      if (state.vgprs[kGlobalAtomicF32AddressVgpr][lane] !=
          static_cast<std::uint32_t>(lane * 4u)) {
        return false;
      }
      for (std::size_t case_index = 0; case_index < kGlobalAtomicF32Cases.size();
           ++case_index) {
        const GlobalAtomicF32Case& atomic_case = kGlobalAtomicF32Cases[case_index];
        const std::uint32_t expected_dst =
            IsGlobalAtomicLaneActive(lane)
                ? InitialGlobalAtomicF32OldValue(case_index, lane)
                : GlobalAtomicF32Sentinel(case_index, lane);
        if (state.vgprs[atomic_case.dst][lane] != expected_dst ||
            state.vgprs[atomic_case.data][lane] !=
                InitialGlobalAtomicF32DataValue(case_index, lane)) {
          return false;
        }
      }
    }
    return true;
  };

  auto initialize_atomic_memory = [](LinearExecutionMemory* memory) {
    for (std::size_t case_index = 0; case_index < kGlobalAtomicF32Cases.size();
         ++case_index) {
      const GlobalAtomicF32Case& atomic_case = kGlobalAtomicF32Cases[case_index];
      for (std::size_t lane = 0; lane < 32u; ++lane) {
        const std::uint64_t address =
            kGlobalAtomicF32BaseAddress + atomic_case.offset + lane * 4u;
        if (!memory->StoreU32(address,
                              InitialGlobalAtomicF32OldValue(case_index, lane))) {
          return false;
        }
      }
    }
    return true;
  };

  auto expect_atomic_memory = [](LinearExecutionMemory* memory) {
    for (std::size_t case_index = 0; case_index < kGlobalAtomicF32Cases.size();
         ++case_index) {
      const GlobalAtomicF32Case& atomic_case = kGlobalAtomicF32Cases[case_index];
      for (std::size_t lane = 0; lane < 32u; ++lane) {
        const std::uint64_t address =
            kGlobalAtomicF32BaseAddress + atomic_case.offset + lane * 4u;
        std::uint32_t value = 0;
        if (!memory->LoadU32(address, &value)) {
          return false;
        }
        const std::uint32_t old_value =
            InitialGlobalAtomicF32OldValue(case_index, lane);
        const std::uint32_t expected_value =
            IsGlobalAtomicLaneActive(lane)
                ? ExpectedGlobalAtomicF32NewValue(
                      case_index, old_value,
                      InitialGlobalAtomicF32DataValue(case_index, lane))
                : old_value;
        if (value != expected_value) {
          return false;
        }
      }
    }
    return true;
  };

  LinearExecutionMemory decoded_atomic_memory(0x1000u, kGlobalAtomicF32BaseAddress);
  if (!Expect(initialize_atomic_memory(&decoded_atomic_memory),
              "expected GLOBAL atomic F32 decoded memory initialization")) {
    return false;
  }
  WaveExecutionState decoded_atomic_state;
  initialize_atomic_state(&decoded_atomic_state);
  if (!Expect(interpreter.ExecuteProgram(atomic_program, &decoded_atomic_state,
                                         &decoded_atomic_memory, error_message),
              "expected decoded GLOBAL atomic F32 execution success") ||
      !Expect(expect_atomic_state(decoded_atomic_state),
              "expected decoded GLOBAL atomic F32 state") ||
      !Expect(expect_atomic_memory(&decoded_atomic_memory),
              "expected decoded GLOBAL atomic F32 memory state")) {
    return false;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_atomic_program;
  if (!Expect(interpreter.CompileProgram(atomic_program, &compiled_atomic_program,
                                         error_message),
              "expected compiled GLOBAL atomic F32 program success") ||
      !Expect(compiled_atomic_program.size() == kGlobalAtomicF32Cases.size() + 1u,
              "expected compiled GLOBAL atomic F32 instruction count")) {
    return false;
  }

  for (std::size_t i = 0; i < kGlobalAtomicF32Cases.size(); ++i) {
    if (!Expect(compiled_atomic_program[i].opcode ==
                    kGlobalAtomicF32Cases[i].compiled,
                "expected compiled GLOBAL atomic F32 opcode order")) {
      return false;
    }
  }
  if (!Expect(compiled_atomic_program.back().opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after GLOBAL atomic F32 batch")) {
    return false;
  }

  LinearExecutionMemory compiled_atomic_memory(0x1000u,
                                               kGlobalAtomicF32BaseAddress);
  if (!Expect(initialize_atomic_memory(&compiled_atomic_memory),
              "expected GLOBAL atomic F32 compiled memory initialization")) {
    return false;
  }
  WaveExecutionState compiled_atomic_state;
  initialize_atomic_state(&compiled_atomic_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_atomic_program,
                                         &compiled_atomic_state,
                                         &compiled_atomic_memory,
                                         error_message),
              "expected compiled GLOBAL atomic F32 execution success") ||
      !Expect(expect_atomic_state(compiled_atomic_state),
              "expected compiled GLOBAL atomic F32 state") ||
      !Expect(expect_atomic_memory(&compiled_atomic_memory),
              "expected compiled GLOBAL atomic F32 memory state")) {
    return false;
  }

  return true;
}

std::uint32_t GlobalAtomicPackedSentinel(std::size_t case_index,
                                         std::size_t lane) {
  return 0xed000000u + static_cast<std::uint32_t>(case_index << 8) +
         static_cast<std::uint32_t>(lane);
}

std::uint32_t InitialGlobalAtomicPackedOldValue(std::size_t case_index,
                                                std::size_t lane) {
  if (!IsGlobalAtomicLaneActive(lane)) {
    return 0xf4000000u + static_cast<std::uint32_t>(case_index << 8) +
           static_cast<std::uint32_t>(lane);
  }

  switch (case_index) {
    case 0:
      return lane == 0u ? PackF16Pair(1.0f, 2.0f)
                        : (lane == 2u ? PackF16Pair(-4.0f, 0.5f)
                                       : PackF16Pair(7.0f, -1.5f));
    case 1:
      return lane == 0u ? PackBf16Pair(1.0f, 2.0f)
                        : (lane == 2u ? PackBf16Pair(-4.0f, 0.5f)
                                       : PackBf16Pair(7.0f, -1.5f));
    default:
      return 0u;
  }
}

std::uint32_t InitialGlobalAtomicPackedDataValue(std::size_t case_index,
                                                 std::size_t lane) {
  if (!IsGlobalAtomicLaneActive(lane)) {
    return 0x54000000u + static_cast<std::uint32_t>(case_index << 8) +
           static_cast<std::uint32_t>(lane);
  }

  switch (case_index) {
    case 0:
      return lane == 0u ? PackF16Pair(0.5f, -1.0f)
                        : (lane == 2u ? PackF16Pair(3.0f, -0.25f)
                                       : PackF16Pair(-1.0f, 2.0f));
    case 1:
      return lane == 0u ? PackBf16Pair(0.5f, -1.0f)
                        : (lane == 2u ? PackBf16Pair(3.0f, -0.25f)
                                       : PackBf16Pair(-1.0f, 2.0f));
    default:
      return 0u;
  }
}

std::uint32_t ExpectedGlobalAtomicPackedNewValue(std::size_t case_index,
                                                 std::uint32_t old_value,
                                                 std::uint32_t data_value) {
  using mirage::sim::isa::PackedBFloat16Add;
  using mirage::sim::isa::PackedHalfAdd;

  switch (case_index) {
    case 0:
      return PackedHalfAdd(old_value, data_value);
    case 1:
      return PackedBFloat16Add(old_value, data_value);
    default:
      return old_value;
  }
}

bool RunGlobalAtomicPackedBatchTest(
    const mirage::sim::isa::Gfx1201BinaryDecoder& decoder,
    const mirage::sim::isa::Gfx1201Interpreter& interpreter,
    std::string* error_message) {
  using namespace mirage::sim::isa;

  std::vector<std::uint32_t> atomic_program_words;
  atomic_program_words.reserve(kGlobalAtomicPackedCases.size() * 2u + 1u);
  for (const GlobalAtomicPackedCase& atomic_case : kGlobalAtomicPackedCases) {
    const auto words =
        MakeGlobal(atomic_case.op, atomic_case.dst,
                   kGlobalAtomicPackedAddressVgpr, atomic_case.data,
                   kGlobalAtomicPackedBaseSgpr, atomic_case.offset);
    atomic_program_words.push_back(words[0]);
    atomic_program_words.push_back(words[1]);
  }
  atomic_program_words.push_back(MakeSopp(48u));

  std::vector<DecodedInstruction> atomic_program;
  if (!Expect(decoder.DecodeProgram(atomic_program_words, &atomic_program,
                                    error_message),
              "expected GLOBAL atomic packed program decode success") ||
      !Expect(atomic_program.size() == kGlobalAtomicPackedCases.size() + 1u,
              "expected decoded GLOBAL atomic packed instruction count")) {
    return false;
  }

  for (std::size_t i = 0; i < kGlobalAtomicPackedCases.size(); ++i) {
    if (!Expect(atomic_program[i].opcode == kGlobalAtomicPackedCases[i].opcode,
                "expected decoded GLOBAL atomic packed opcode order")) {
      return false;
    }
  }
  if (!Expect(atomic_program.back().opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after GLOBAL atomic packed batch")) {
    return false;
  }

  auto initialize_atomic_state = [](WaveExecutionState* state) {
    state->exec_mask = kGlobalAtomicExecMask;
    state->sgprs[kGlobalAtomicPackedBaseSgpr] =
        static_cast<std::uint32_t>(kGlobalAtomicPackedBaseAddress);
    state->sgprs[kGlobalAtomicPackedBaseSgpr + 1u] = 0u;

    for (std::size_t lane = 0; lane < 32u; ++lane) {
      state->vgprs[kGlobalAtomicPackedAddressVgpr][lane] =
          static_cast<std::uint32_t>(lane * 4u);
      for (std::size_t case_index = 0;
           case_index < kGlobalAtomicPackedCases.size(); ++case_index) {
        const GlobalAtomicPackedCase& atomic_case =
            kGlobalAtomicPackedCases[case_index];
        state->vgprs[atomic_case.dst][lane] =
            GlobalAtomicPackedSentinel(case_index, lane);
        state->vgprs[atomic_case.data][lane] =
            InitialGlobalAtomicPackedDataValue(case_index, lane);
      }
    }
  };

  auto expect_atomic_state = [](const WaveExecutionState& state) {
    if (!(state.lane_count == 32u && state.exec_mask == kGlobalAtomicExecMask &&
          state.sgprs[kGlobalAtomicPackedBaseSgpr] ==
              static_cast<std::uint32_t>(kGlobalAtomicPackedBaseAddress) &&
          state.sgprs[kGlobalAtomicPackedBaseSgpr + 1u] == 0u &&
          state.halted && !state.waiting_on_barrier &&
          state.pc == kGlobalAtomicPackedCases.size())) {
      return false;
    }

    for (std::size_t lane = 0; lane < 32u; ++lane) {
      if (state.vgprs[kGlobalAtomicPackedAddressVgpr][lane] !=
          static_cast<std::uint32_t>(lane * 4u)) {
        return false;
      }
      for (std::size_t case_index = 0;
           case_index < kGlobalAtomicPackedCases.size(); ++case_index) {
        const GlobalAtomicPackedCase& atomic_case =
            kGlobalAtomicPackedCases[case_index];
        const std::uint32_t expected_dst =
            IsGlobalAtomicLaneActive(lane)
                ? InitialGlobalAtomicPackedOldValue(case_index, lane)
                : GlobalAtomicPackedSentinel(case_index, lane);
        if (state.vgprs[atomic_case.dst][lane] != expected_dst ||
            state.vgprs[atomic_case.data][lane] !=
                InitialGlobalAtomicPackedDataValue(case_index, lane)) {
          return false;
        }
      }
    }
    return true;
  };

  auto initialize_atomic_memory = [](LinearExecutionMemory* memory) {
    for (std::size_t case_index = 0;
         case_index < kGlobalAtomicPackedCases.size(); ++case_index) {
      const GlobalAtomicPackedCase& atomic_case =
          kGlobalAtomicPackedCases[case_index];
      for (std::size_t lane = 0; lane < 32u; ++lane) {
        const std::uint64_t address =
            kGlobalAtomicPackedBaseAddress + atomic_case.offset + lane * 4u;
        if (!memory->StoreU32(
                address, InitialGlobalAtomicPackedOldValue(case_index, lane))) {
          return false;
        }
      }
    }
    return true;
  };

  auto expect_atomic_memory = [](LinearExecutionMemory* memory) {
    for (std::size_t case_index = 0;
         case_index < kGlobalAtomicPackedCases.size(); ++case_index) {
      const GlobalAtomicPackedCase& atomic_case =
          kGlobalAtomicPackedCases[case_index];
      for (std::size_t lane = 0; lane < 32u; ++lane) {
        const std::uint64_t address =
            kGlobalAtomicPackedBaseAddress + atomic_case.offset + lane * 4u;
        std::uint32_t value = 0;
        if (!memory->LoadU32(address, &value)) {
          return false;
        }
        const std::uint32_t old_value =
            InitialGlobalAtomicPackedOldValue(case_index, lane);
        const std::uint32_t expected_value =
            IsGlobalAtomicLaneActive(lane)
                ? ExpectedGlobalAtomicPackedNewValue(
                      case_index, old_value,
                      InitialGlobalAtomicPackedDataValue(case_index, lane))
                : old_value;
        if (value != expected_value) {
          return false;
        }
      }
    }
    return true;
  };

  LinearExecutionMemory decoded_atomic_memory(0x1000u,
                                              kGlobalAtomicPackedBaseAddress);
  if (!Expect(initialize_atomic_memory(&decoded_atomic_memory),
              "expected GLOBAL atomic packed decoded memory initialization")) {
    return false;
  }
  WaveExecutionState decoded_atomic_state;
  initialize_atomic_state(&decoded_atomic_state);
  if (!Expect(interpreter.ExecuteProgram(atomic_program, &decoded_atomic_state,
                                         &decoded_atomic_memory, error_message),
              "expected decoded GLOBAL atomic packed execution success") ||
      !Expect(expect_atomic_state(decoded_atomic_state),
              "expected decoded GLOBAL atomic packed state") ||
      !Expect(expect_atomic_memory(&decoded_atomic_memory),
              "expected decoded GLOBAL atomic packed memory state")) {
    return false;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_atomic_program;
  if (!Expect(interpreter.CompileProgram(atomic_program, &compiled_atomic_program,
                                         error_message),
              "expected compiled GLOBAL atomic packed program success") ||
      !Expect(compiled_atomic_program.size() ==
                  kGlobalAtomicPackedCases.size() + 1u,
              "expected compiled GLOBAL atomic packed instruction count")) {
    return false;
  }

  for (std::size_t i = 0; i < kGlobalAtomicPackedCases.size(); ++i) {
    if (!Expect(compiled_atomic_program[i].opcode ==
                    kGlobalAtomicPackedCases[i].compiled,
                "expected compiled GLOBAL atomic packed opcode order")) {
      return false;
    }
  }
  if (!Expect(compiled_atomic_program.back().opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after GLOBAL atomic packed batch")) {
    return false;
  }

  LinearExecutionMemory compiled_atomic_memory(0x1000u,
                                               kGlobalAtomicPackedBaseAddress);
  if (!Expect(initialize_atomic_memory(&compiled_atomic_memory),
              "expected GLOBAL atomic packed compiled memory initialization")) {
    return false;
  }
  WaveExecutionState compiled_atomic_state;
  initialize_atomic_state(&compiled_atomic_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_atomic_program,
                                         &compiled_atomic_state,
                                         &compiled_atomic_memory,
                                         error_message),
              "expected compiled GLOBAL atomic packed execution success") ||
      !Expect(expect_atomic_state(compiled_atomic_state),
              "expected compiled GLOBAL atomic packed state") ||
      !Expect(expect_atomic_memory(&compiled_atomic_memory),
              "expected compiled GLOBAL atomic packed memory state")) {
    return false;
  }

  return true;
}

constexpr std::uint64_t kDsExecMask = 0xbu;
constexpr std::uint16_t kDsAddressVgprBase = 64u;
constexpr std::uint64_t kDsBaseAddress = 0x12000u;

struct DsCase {
  const char* opcode;
  std::uint32_t op;
  mirage::sim::isa::Gfx1201CompiledOpcode compiled;
  std::uint16_t addr;
  std::uint16_t data;
  std::uint32_t offset;
};

constexpr std::array<DsCase, 13> kDsCases{{
    {"DS_ADD_F32", 21u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kDsAddF32, 64u, 80u, 0x000u},
    {"DS_ADD_U32", 0u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kDsAddU32, 65u, 81u, 0x010u},
    {"DS_SUB_U32", 1u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kDsSubU32, 66u, 82u, 0x020u},
    {"DS_RSUB_U32", 2u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kDsRsubU32, 67u, 83u, 0x030u},
    {"DS_INC_U32", 3u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kDsIncU32, 68u, 84u, 0x040u},
    {"DS_DEC_U32", 4u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kDsDecU32, 69u, 85u, 0x050u},
    {"DS_MIN_I32", 5u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kDsMinI32, 70u, 86u, 0x060u},
    {"DS_MIN_U32", 7u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kDsMinU32, 71u, 87u, 0x070u},
    {"DS_MAX_I32", 6u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kDsMaxI32, 72u, 88u, 0x080u},
    {"DS_MAX_U32", 8u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kDsMaxU32, 73u, 89u, 0x090u},
    {"DS_AND_B32", 9u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kDsAndB32, 74u, 90u, 0x0a0u},
    {"DS_OR_B32", 10u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kDsOrB32, 75u, 91u, 0x0b0u},
    {"DS_XOR_B32", 11u,
     mirage::sim::isa::Gfx1201CompiledOpcode::kDsXorB32, 76u, 92u, 0x0c0u},
}};

std::uint64_t DsCaseBaseAddress(std::size_t case_index) {
  return kDsBaseAddress + static_cast<std::uint64_t>(case_index) * 0x100u;
}

bool IsDsLaneActive(std::size_t lane) {
  return (kDsExecMask & (1ull << lane)) != 0u;
}

std::uint32_t InitialDsOldValue(std::size_t case_index, std::size_t lane) {
  if (!IsDsLaneActive(lane)) {
    return 0xf5000000u + static_cast<std::uint32_t>(case_index << 8) +
           static_cast<std::uint32_t>(lane);
  }

  switch (case_index) {
    case 0:
      return lane == 0u ? FloatBits(1.5f)
                        : (lane == 1u ? FloatBits(-4.5f)
                                       : FloatBits(8.0f));
    case 1:
      return lane == 0u ? 10u : (lane == 1u ? 100u : 0xfffffff0u);
    case 2:
      return lane == 0u ? 20u : (lane == 1u ? 5u : 1u);
    case 3:
      return lane == 0u ? 4u : (lane == 1u ? 10u : 1u);
    case 4:
      return lane == 0u ? 4u : (lane == 1u ? 3u : 8u);
    case 5:
      return lane == 0u ? 0u : (lane == 1u ? 5u : 4u);
    case 6:
      return lane == 0u ? I32Bits(-5) : (lane == 1u ? 9u : I32Bits(-1));
    case 7:
      return lane == 0u ? 5u : (lane == 1u ? 9u : 0xfffffff0u);
    case 8:
      return lane == 0u ? I32Bits(-5) : (lane == 1u ? 9u : I32Bits(-1));
    case 9:
      return lane == 0u ? 5u : (lane == 1u ? 9u : 0xfffffff0u);
    case 10:
      return lane == 0u ? 0x0f0f00f0u
                        : (lane == 1u ? 0xff00ff00u : 0xaaaaaaaau);
    case 11:
      return lane == 0u ? 0x000000f0u
                        : (lane == 1u ? 0x00ff0000u : 0x0f000f00u);
    case 12:
      return lane == 0u ? 0xaaaa5555u
                        : (lane == 1u ? 0x12345678u : 0xffffffffu);
    default:
      return 0u;
  }
}

std::uint32_t InitialDsDataValue(std::size_t case_index, std::size_t lane) {
  if (!IsDsLaneActive(lane)) {
    return 0x56000000u + static_cast<std::uint32_t>(case_index << 8) +
           static_cast<std::uint32_t>(lane);
  }

  switch (case_index) {
    case 0:
      return lane == 0u ? FloatBits(2.25f)
                        : (lane == 1u ? FloatBits(1.0f)
                                       : FloatBits(-1.5f));
    case 1:
      return lane == 0u ? 5u : (lane == 1u ? 7u : 0x20u);
    case 2:
      return lane == 0u ? 3u : (lane == 1u ? 7u : 2u);
    case 3:
      return lane == 0u ? 7u : (lane == 1u ? 6u : 1u);
    case 4:
      return 4u;
    case 5:
      return 4u;
    case 6:
      return lane == 0u ? I32Bits(-2) : (lane == 1u ? I32Bits(-4) : 7u);
    case 7:
      return lane == 0u ? 2u : (lane == 1u ? 10u : 0x10u);
    case 8:
      return lane == 0u ? I32Bits(-2) : (lane == 1u ? I32Bits(-4) : 7u);
    case 9:
      return lane == 0u ? 2u : (lane == 1u ? 10u : 0x10u);
    case 10:
      return lane == 0u ? 0x00ff0ff0u
                        : (lane == 1u ? 0x0f0f0f0fu : 0x00ffff00u);
    case 11:
      return lane == 0u ? 0x00000f00u
                        : (lane == 1u ? 0x000000ffu : 0xf000000fu);
    case 12:
      return lane == 0u ? 0xffff0000u
                        : (lane == 1u ? 0x0f0f0f0fu : 0x12345678u);
    default:
      return 0u;
  }
}

std::uint32_t ExpectedDsNewValue(std::size_t case_index,
                                 std::uint32_t old_value,
                                 std::uint32_t data_value) {
  switch (case_index) {
    case 0:
      return FloatBits(BitsToFloat(old_value) + BitsToFloat(data_value));
    case 1:
      return old_value + data_value;
    case 2:
      return old_value - data_value;
    case 3:
      return data_value - old_value;
    case 4:
      return old_value >= data_value ? 0u : old_value + 1u;
    case 5:
      return (old_value == 0u || old_value > data_value) ? data_value
                                                          : old_value - 1u;
    case 6:
      return I32Bits(std::min(BitsToI32(old_value), BitsToI32(data_value)));
    case 7:
      return std::min(old_value, data_value);
    case 8:
      return I32Bits(std::max(BitsToI32(old_value), BitsToI32(data_value)));
    case 9:
      return std::max(old_value, data_value);
    case 10:
      return old_value & data_value;
    case 11:
      return old_value | data_value;
    case 12:
      return old_value ^ data_value;
    default:
      return old_value;
  }
}

bool RunDsBatchTest(const mirage::sim::isa::Gfx1201BinaryDecoder& decoder,
                    const mirage::sim::isa::Gfx1201Interpreter& interpreter,
                    std::string* error_message) {
  using namespace mirage::sim::isa;

  std::vector<std::uint32_t> ds_program_words;
  ds_program_words.reserve((kDsCases.size() + 1u) * 2u + 1u);
  const auto ds_nop_words = MakeDs(20u, 0u, 0u, 0u, 0u, 0u);
  ds_program_words.push_back(ds_nop_words[0]);
  ds_program_words.push_back(ds_nop_words[1]);
  for (const DsCase& ds_case : kDsCases) {
    const auto words =
        MakeDs(ds_case.op, 0u, ds_case.addr, ds_case.data, 0u, ds_case.offset);
    ds_program_words.push_back(words[0]);
    ds_program_words.push_back(words[1]);
  }
  ds_program_words.push_back(MakeSopp(48u));

  std::vector<DecodedInstruction> ds_program;
  if (!Expect(decoder.DecodeProgram(ds_program_words, &ds_program, error_message),
              "expected DS batch decode success") ||
      !Expect(ds_program.size() == kDsCases.size() + 2u,
              "expected decoded DS instruction count") ||
      !Expect(ds_program.front().opcode == "DS_NOP",
              "expected decoded DS_NOP at batch start")) {
    return false;
  }
  for (std::size_t i = 0; i < kDsCases.size(); ++i) {
    if (!Expect(ds_program[i + 1u].opcode == kDsCases[i].opcode,
                "expected decoded DS opcode order")) {
      return false;
    }
  }
  if (!Expect(ds_program.back().opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after DS batch")) {
    return false;
  }

  auto initialize_ds_state = [](WaveExecutionState* state) {
    state->exec_mask = kDsExecMask;
    for (std::size_t lane = 0; lane < 32u; ++lane) {
      for (std::size_t case_index = 0; case_index < kDsCases.size();
           ++case_index) {
        state->vgprs[kDsCases[case_index].addr][lane] = static_cast<std::uint32_t>(
            DsCaseBaseAddress(case_index) + lane * 4u);
        state->vgprs[kDsCases[case_index].data][lane] =
            InitialDsDataValue(case_index, lane);
      }
    }
  };

  auto expect_ds_state = [](const WaveExecutionState& state) {
    if (!(state.lane_count == 32u && state.exec_mask == kDsExecMask &&
          state.halted && !state.waiting_on_barrier &&
          state.pc == kDsCases.size() + 1u)) {
      return false;
    }
    for (std::size_t lane = 0; lane < 32u; ++lane) {
      for (std::size_t case_index = 0; case_index < kDsCases.size();
           ++case_index) {
        if (state.vgprs[kDsCases[case_index].addr][lane] !=
            static_cast<std::uint32_t>(DsCaseBaseAddress(case_index) +
                                       lane * 4u)) {
          return false;
        }
        if (state.vgprs[kDsCases[case_index].data][lane] !=
            InitialDsDataValue(case_index, lane)) {
          return false;
        }
      }
    }
    return true;
  };

  auto initialize_ds_memory = [](LinearExecutionMemory* memory) {
    for (std::size_t case_index = 0; case_index < kDsCases.size();
         ++case_index) {
      for (std::size_t lane = 0; lane < 32u; ++lane) {
        const std::uint64_t address =
            DsCaseBaseAddress(case_index) + kDsCases[case_index].offset +
            lane * 4u;
        if (!memory->StoreU32(address, InitialDsOldValue(case_index, lane))) {
          return false;
        }
      }
    }
    return true;
  };

  auto expect_ds_memory = [](LinearExecutionMemory* memory) {
    for (std::size_t case_index = 0; case_index < kDsCases.size();
         ++case_index) {
      for (std::size_t lane = 0; lane < 32u; ++lane) {
        const std::uint64_t address =
            DsCaseBaseAddress(case_index) + kDsCases[case_index].offset +
            lane * 4u;
        std::uint32_t value = 0;
        if (!memory->LoadU32(address, &value)) {
          return false;
        }
        const std::uint32_t old_value = InitialDsOldValue(case_index, lane);
        const std::uint32_t expected_value =
            IsDsLaneActive(lane)
                ? ExpectedDsNewValue(case_index, old_value,
                                     InitialDsDataValue(case_index, lane))
                : old_value;
        if (value != expected_value) {
          return false;
        }
      }
    }
    return true;
  };

  LinearExecutionMemory decoded_ds_memory(0x2000u, kDsBaseAddress);
  if (!Expect(initialize_ds_memory(&decoded_ds_memory),
              "expected DS decoded memory initialization")) {
    return false;
  }
  WaveExecutionState decoded_ds_state;
  initialize_ds_state(&decoded_ds_state);
  if (!Expect(interpreter.ExecuteProgram(ds_program, &decoded_ds_state,
                                         &decoded_ds_memory, error_message),
              "expected decoded DS execution success") ||
      !Expect(expect_ds_state(decoded_ds_state),
              "expected decoded DS state") ||
      !Expect(expect_ds_memory(&decoded_ds_memory),
              "expected decoded DS memory state")) {
    return false;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_ds_program;
  if (!Expect(interpreter.CompileProgram(ds_program, &compiled_ds_program,
                                         error_message),
              "expected compiled DS program success") ||
      !Expect(compiled_ds_program.size() == kDsCases.size() + 2u,
              "expected compiled DS instruction count") ||
      !Expect(compiled_ds_program.front().opcode == Gfx1201CompiledOpcode::kSNop,
              "expected compiled DS_NOP as kSNop")) {
    return false;
  }
  for (std::size_t i = 0; i < kDsCases.size(); ++i) {
    if (!Expect(compiled_ds_program[i + 1u].opcode == kDsCases[i].compiled,
                "expected compiled DS opcode order")) {
      return false;
    }
  }
  if (!Expect(compiled_ds_program.back().opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after DS batch")) {
    return false;
  }

  LinearExecutionMemory compiled_ds_memory(0x2000u, kDsBaseAddress);
  if (!Expect(initialize_ds_memory(&compiled_ds_memory),
              "expected DS compiled memory initialization")) {
    return false;
  }
  WaveExecutionState compiled_ds_state;
  initialize_ds_state(&compiled_ds_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_ds_program,
                                         &compiled_ds_state,
                                         &compiled_ds_memory,
                                         error_message),
              "expected compiled DS execution success") ||
      !Expect(expect_ds_state(compiled_ds_state),
              "expected compiled DS state") ||
      !Expect(expect_ds_memory(&compiled_ds_memory),
              "expected compiled DS memory state")) {
    return false;
  }

  return true;
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

bool ExpectF16ClassCndmaskState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.vgprs[60][0] == 20u && state.vgprs[60][1] == 10u &&
         state.vgprs[60][2] == 20u && state.vgprs[60][3] == 10u &&
         state.vcc_mask == 5u && state.exec_mask == 0xfu && state.halted &&
         !state.waiting_on_barrier && state.pc == 3u;
}

bool ExpectF16CmpxClassBranchState(
    const mirage::sim::isa::WaveExecutionState& state) {
  return state.sgprs[114] == kQuietNaNF16Bits && state.sgprs[119] == 222u &&
         state.vcc_mask == 10u && state.exec_mask == 10u && state.halted &&
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

  const std::array<std::uint32_t, 1> v_nop_words{MakeVop1(0u, 0u, 0u)};
  if (!Expect(decoder.DecodeInstruction(v_nop_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_NOP decode success") ||
      !Expect(words_consumed == 1u, "expected one dword consumed for V_NOP") ||
      !Expect(instruction.opcode == "V_NOP", "expected V_NOP opcode") ||
      !Expect(instruction.operand_count == 0u,
              "expected V_NOP nullary decode")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_pipeflush_words{MakeVop1(27u, 0u, 0u)};
  if (!Expect(decoder.DecodeInstruction(v_pipeflush_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_PIPEFLUSH decode success") ||
      !Expect(words_consumed == 1u,
              "expected one dword consumed for V_PIPEFLUSH") ||
      !Expect(instruction.opcode == "V_PIPEFLUSH",
              "expected V_PIPEFLUSH opcode") ||
      !Expect(instruction.operand_count == 0u,
              "expected V_PIPEFLUSH nullary decode")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_readfirstlane_words{
      MakeVop1(2u, 22u, 278u)};
  if (!Expect(decoder.DecodeInstruction(v_readfirstlane_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_READFIRSTLANE_B32 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_READFIRSTLANE_B32",
                                     OperandKind::kSgpr, 22u,
                                     OperandKind::kVgpr, 22u),
              "expected decoded V_READFIRSTLANE_B32 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kScalarDestination,
                  OperandValueClass::kScalarRegister, OperandAccess::kWrite,
                  FragmentKind::kScalar, 32u, 1u, false),
              "expected V_READFIRSTLANE_B32 destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kVectorRegister, OperandAccess::kRead,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_READFIRSTLANE_B32 source descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_movreld_words{MakeVop1(66u, 40u, 5u)};
  if (!Expect(decoder.DecodeInstruction(v_movreld_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_MOVRELD_B32 decode success") ||
      !Expect(ExpectThreeOperandInstruction(
                  instruction, "V_MOVRELD_B32", OperandKind::kVgpr, 40u,
                  OperandKind::kSgpr, 5u, OperandKind::kSgpr, kM0RegisterIndex),
              "expected decoded V_MOVRELD_B32 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kVectorRegister, OperandAccess::kWrite,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_MOVRELD_B32 destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 32u, 1u, false),
              "expected V_MOVRELD_B32 source descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[2], OperandRole::kSource1,
                  OperandSlotKind::kSource1,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 32u, 1u, true),
              "expected V_MOVRELD_B32 implicit M0 descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_movrels_words{MakeVop1(67u, 41u, 276u)};
  if (!Expect(decoder.DecodeInstruction(v_movrels_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_MOVRELS_B32 decode success") ||
      !Expect(ExpectThreeOperandInstruction(
                  instruction, "V_MOVRELS_B32", OperandKind::kVgpr, 41u,
                  OperandKind::kVgpr, 20u, OperandKind::kSgpr, kM0RegisterIndex),
              "expected decoded V_MOVRELS_B32 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[2], OperandRole::kSource1,
                  OperandSlotKind::kSource1,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 32u, 1u, true),
              "expected V_MOVRELS_B32 implicit M0 descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_movrelsd_words{MakeVop1(68u, 42u, 277u)};
  if (!Expect(decoder.DecodeInstruction(v_movrelsd_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_MOVRELSD_B32 decode success") ||
      !Expect(ExpectThreeOperandInstruction(
                  instruction, "V_MOVRELSD_B32", OperandKind::kVgpr, 42u,
                  OperandKind::kVgpr, 21u, OperandKind::kSgpr, kM0RegisterIndex),
              "expected decoded V_MOVRELSD_B32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_movrelsd2_words{MakeVop1(72u, 43u, 278u)};
  if (!Expect(decoder.DecodeInstruction(v_movrelsd2_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_MOVRELSD_2_B32 decode success") ||
      !Expect(ExpectThreeOperandInstruction(
                  instruction, "V_MOVRELSD_2_B32", OperandKind::kVgpr, 43u,
                  OperandKind::kVgpr, 22u, OperandKind::kSgpr, kM0RegisterIndex),
              "expected decoded V_MOVRELSD_2_B32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_swaprel_words{MakeVop1(104u, 44u, 279u)};
  if (!Expect(decoder.DecodeInstruction(v_swaprel_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_SWAPREL_B32 decode success") ||
      !Expect(ExpectThreeOperandInstruction(
                  instruction, "V_SWAPREL_B32", OperandKind::kVgpr, 44u,
                  OperandKind::kVgpr, 23u, OperandKind::kSgpr, kM0RegisterIndex),
              "expected decoded V_SWAPREL_B32 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kVectorRegister,
                  OperandAccess::kReadWrite, FragmentKind::kVector, 32u, 1u,
                  false),
              "expected V_SWAPREL_B32 destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kVectorRegister,
                  OperandAccess::kReadWrite, FragmentKind::kVector, 32u, 1u,
                  false),
              "expected V_SWAPREL_B32 source descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[2], OperandRole::kSource1,
                  OperandSlotKind::kSource1,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 32u, 1u, true),
              "expected V_SWAPREL_B32 implicit M0 descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_mov_b16_words{MakeVop1(28u, 30u, 257u)};
  if (!Expect(decoder.DecodeInstruction(v_mov_b16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_MOV_B16 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_MOV_B16",
                                     OperandKind::kVgpr, 30u,
                                     OperandKind::kVgpr, 1u),
              "expected decoded V_MOV_B16 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_permlane64_b32_words{
      MakeVop1(103u, 31u, 260u)};
  if (!Expect(decoder.DecodeInstruction(v_permlane64_b32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_PERMLANE64_B32 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_PERMLANE64_B32",
                                     OperandKind::kVgpr, 31u,
                                     OperandKind::kVgpr, 4u),
              "expected decoded V_PERMLANE64_B32 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kVectorRegister, OperandAccess::kWrite,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_PERMLANE64_B32 destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kVectorRegister, OperandAccess::kRead,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_PERMLANE64_B32 source descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_swap_b32_words{
      MakeVop1(101u, 33u, 290u)};
  if (!Expect(decoder.DecodeInstruction(v_swap_b32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_SWAP_B32 decode success") ||
      !Expect(instruction.opcode == "V_SWAP_B32",
              "expected V_SWAP_B32 opcode") ||
      !Expect(instruction.operand_count == 2u,
              "expected V_SWAP_B32 two-operand decode") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kVectorRegister,
                  OperandAccess::kReadWrite, FragmentKind::kVector, 32u, 1u,
                  false),
              "expected V_SWAP_B32 destination readwrite descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kVectorRegister,
                  OperandAccess::kReadWrite, FragmentKind::kVector, 32u, 1u,
                  false),
              "expected V_SWAP_B32 source readwrite descriptor") ||
      !Expect(instruction.operands[0].kind == OperandKind::kVgpr &&
                  instruction.operands[0].index == 33u &&
                  instruction.operands[1].kind == OperandKind::kVgpr &&
                  instruction.operands[1].index == 34u,
              "expected decoded V_SWAP_B32 registers")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_swap_b16_words{
      MakeVop1(102u, 35u, 292u)};
  if (!Expect(decoder.DecodeInstruction(v_swap_b16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_SWAP_B16 decode success") ||
      !Expect(instruction.opcode == "V_SWAP_B16",
              "expected V_SWAP_B16 opcode") ||
      !Expect(instruction.operand_count == 2u,
              "expected V_SWAP_B16 two-operand decode") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kVectorRegister,
                  OperandAccess::kReadWrite, FragmentKind::kVector, 16u, 1u,
                  false),
              "expected V_SWAP_B16 destination readwrite descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kVectorRegister,
                  OperandAccess::kReadWrite, FragmentKind::kVector, 16u, 1u,
                  false),
              "expected V_SWAP_B16 source readwrite descriptor") ||
      !Expect(instruction.operands[0].kind == OperandKind::kVgpr &&
                  instruction.operands[0].index == 35u &&
                  instruction.operands[1].kind == OperandKind::kVgpr &&
                  instruction.operands[1].index == 36u,
              "expected decoded V_SWAP_B16 registers")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_not_b16_words{MakeVop1(105u, 32u, 2u)};
  if (!Expect(decoder.DecodeInstruction(v_not_b16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_NOT_B16 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_NOT_B16",
                                     OperandKind::kVgpr, 32u,
                                     OperandKind::kSgpr, 2u),
              "expected decoded V_NOT_B16 operands")) {
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

  const std::array<std::uint32_t, 1> v_clz_i32_u32_words{
      MakeVop1(57u, 18u, 257u)};
  if (!Expect(decoder.DecodeInstruction(v_clz_i32_u32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CLZ_I32_U32 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CLZ_I32_U32",
                                     OperandKind::kVgpr, 18u,
                                     OperandKind::kVgpr, 1u),
              "expected decoded V_CLZ_I32_U32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cls_i32_words{MakeVop1(59u, 19u, 258u)};
  if (!Expect(decoder.DecodeInstruction(v_cls_i32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CLS_I32 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CLS_I32",
                                     OperandKind::kVgpr, 19u,
                                     OperandKind::kVgpr, 2u),
              "expected decoded V_CLS_I32 operands")) {
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

  const std::array<std::uint32_t, 1> v_cvt_f32_fp8_words{
      MakeVop1(108u, 30u, 257u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_f32_fp8_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_F32_FP8 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_F32_FP8",
                                     OperandKind::kVgpr, 30u,
                                     OperandKind::kVgpr, 1u),
              "expected decoded V_CVT_F32_FP8 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kVectorRegister, OperandAccess::kWrite,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_CVT_F32_FP8 destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kVectorRegister, OperandAccess::kRead,
                  FragmentKind::kVector, 8u, 1u, false),
              "expected V_CVT_F32_FP8 source descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_f32_bf8_words{
      MakeVop1(109u, 31u, 3u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_f32_bf8_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_F32_BF8 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_F32_BF8",
                                     OperandKind::kVgpr, 31u,
                                     OperandKind::kSgpr, 3u),
              "expected decoded V_CVT_F32_BF8 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kVectorRegister, OperandAccess::kWrite,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_CVT_F32_BF8 destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 8u, 1u, false),
              "expected V_CVT_F32_BF8 source descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_pk_f32_fp8_words{
      MakeVop1(110u, 32u, 257u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_pk_f32_fp8_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_PK_F32_FP8 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_PK_F32_FP8",
                                     OperandKind::kVgpr, 32u,
                                     OperandKind::kVgpr, 1u),
              "expected decoded V_CVT_PK_F32_FP8 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kVectorRegister, OperandAccess::kWrite,
                  FragmentKind::kVector, 32u, 2u, false),
              "expected V_CVT_PK_F32_FP8 destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kPackedVector, OperandAccess::kRead,
                  FragmentKind::kPacked, 8u, 2u, false),
              "expected V_CVT_PK_F32_FP8 source descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_pk_f32_bf8_words{
      MakeVop1(111u, 33u, 3u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_pk_f32_bf8_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_PK_F32_BF8 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_PK_F32_BF8",
                                     OperandKind::kVgpr, 33u,
                                     OperandKind::kSgpr, 3u),
              "expected decoded V_CVT_PK_F32_BF8 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kVectorRegister, OperandAccess::kWrite,
                  FragmentKind::kVector, 32u, 2u, false),
              "expected V_CVT_PK_F32_BF8 destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kPacked, 8u, 2u, false),
              "expected V_CVT_PK_F32_BF8 source descriptor")) {
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

  const std::array<std::uint32_t, 1> v_cvt_nearest_i32_f32_words{
      MakeVop1(12u, 20u, 261u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_nearest_i32_f32_words,
                                        &instruction, &words_consumed,
                                        &error_message),
              "expected V_CVT_NEAREST_I32_F32 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_NEAREST_I32_F32",
                                     OperandKind::kVgpr, 20u,
                                     OperandKind::kVgpr, 5u),
              "expected decoded V_CVT_NEAREST_I32_F32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_floor_i32_f32_words{
      MakeVop1(13u, 21u, 262u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_floor_i32_f32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_FLOOR_I32_F32 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_FLOOR_I32_F32",
                                     OperandKind::kVgpr, 21u,
                                     OperandKind::kVgpr, 6u),
              "expected decoded V_CVT_FLOOR_I32_F32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_i32_i16_words{
      MakeVop1(106u, 22u, 257u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_i32_i16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_I32_I16 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_I32_I16",
                                     OperandKind::kVgpr, 22u,
                                     OperandKind::kVgpr, 1u),
              "expected decoded V_CVT_I32_I16 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_u32_u16_words{
      MakeVop1(107u, 23u, 2u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_u32_u16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_U32_U16 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_U32_U16",
                                     OperandKind::kVgpr, 23u,
                                     OperandKind::kSgpr, 2u),
              "expected decoded V_CVT_U32_U16 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_off_f32_i4_words{
      MakeVop1(14u, 37u, 257u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_off_f32_i4_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_OFF_F32_I4 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_OFF_F32_I4",
                                     OperandKind::kVgpr, 37u,
                                     OperandKind::kVgpr, 1u),
              "expected decoded V_CVT_OFF_F32_I4 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_norm_i16_f16_words{
      MakeVop1(99u, 38u, 258u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_norm_i16_f16_words,
                                        &instruction, &words_consumed,
                                        &error_message),
              "expected V_CVT_NORM_I16_F16 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_NORM_I16_F16",
                                     OperandKind::kVgpr, 38u,
                                     OperandKind::kVgpr, 2u),
              "expected decoded V_CVT_NORM_I16_F16 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_norm_u16_f16_words{
      MakeVop1(100u, 39u, 3u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_norm_u16_f16_words,
                                        &instruction, &words_consumed,
                                        &error_message),
              "expected V_CVT_NORM_U16_F16 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_NORM_U16_F16",
                                     OperandKind::kVgpr, 39u,
                                     OperandKind::kSgpr, 3u),
              "expected decoded V_CVT_NORM_U16_F16 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_f16_f32_words{
      MakeVop1(10u, 24u, 257u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_f16_f32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_F16_F32 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_F16_F32",
                                     OperandKind::kVgpr, 24u,
                                     OperandKind::kVgpr, 1u),
              "expected decoded V_CVT_F16_F32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_f16_i16_words{
      MakeVop1(81u, 25u, 258u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_f16_i16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_F16_I16 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_F16_I16",
                                     OperandKind::kVgpr, 25u,
                                     OperandKind::kVgpr, 2u),
              "expected decoded V_CVT_F16_I16 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_f16_u16_words{
      MakeVop1(80u, 26u, 3u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_f16_u16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_F16_U16 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_F16_U16",
                                     OperandKind::kVgpr, 26u,
                                     OperandKind::kSgpr, 3u),
              "expected decoded V_CVT_F16_U16 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_f32_f16_words{
      MakeVop1(11u, 27u, 259u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_f32_f16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_F32_F16 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_F32_F16",
                                     OperandKind::kVgpr, 27u,
                                     OperandKind::kVgpr, 3u),
              "expected decoded V_CVT_F32_F16 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_i16_f16_words{
      MakeVop1(83u, 28u, 260u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_i16_f16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_I16_F16 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_I16_F16",
                                     OperandKind::kVgpr, 28u,
                                     OperandKind::kVgpr, 4u),
              "expected decoded V_CVT_I16_F16 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cvt_u16_f16_words{
      MakeVop1(82u, 29u, 5u)};
  if (!Expect(decoder.DecodeInstruction(v_cvt_u16_f16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CVT_U16_F16 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_CVT_U16_F16",
                                     OperandKind::kVgpr, 29u,
                                     OperandKind::kSgpr, 5u),
              "expected decoded V_CVT_U16_F16 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_sat_pk_u8_i16_words{
      MakeVop1(98u, 30u, 257u)};
  if (!Expect(decoder.DecodeInstruction(v_sat_pk_u8_i16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_SAT_PK_U8_I16 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_SAT_PK_U8_I16",
                                     OperandKind::kVgpr, 30u,
                                     OperandKind::kVgpr, 1u),
              "expected decoded V_SAT_PK_U8_I16 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_rcp_f16_words{
      MakeVop1(84u, 40u, 257u)};
  if (!Expect(decoder.DecodeInstruction(v_rcp_f16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_RCP_F16 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_RCP_F16",
                                     OperandKind::kVgpr, 40u,
                                     OperandKind::kVgpr, 1u),
              "expected decoded V_RCP_F16 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_frexp_exp_i16_f16_words{
      MakeVop1(90u, 41u, 3u)};
  if (!Expect(decoder.DecodeInstruction(v_frexp_exp_i16_f16_words,
                                        &instruction, &words_consumed,
                                        &error_message),
              "expected V_FREXP_EXP_I16_F16 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_FREXP_EXP_I16_F16",
                                     OperandKind::kVgpr, 41u,
                                     OperandKind::kSgpr, 3u),
              "expected decoded V_FREXP_EXP_I16_F16 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_trunc_f16_words{
      MakeVop1(93u, 42u, 258u)};
  if (!Expect(decoder.DecodeInstruction(v_trunc_f16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_TRUNC_F16 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_TRUNC_F16",
                                     OperandKind::kVgpr, 42u,
                                     OperandKind::kVgpr, 2u),
              "expected decoded V_TRUNC_F16 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cos_f16_words{
      MakeVop1(97u, 43u, 259u)};
  if (!Expect(decoder.DecodeInstruction(v_cos_f16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_COS_F16 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_COS_F16",
                                     OperandKind::kVgpr, 43u,
                                     OperandKind::kVgpr, 3u),
              "expected decoded V_COS_F16 operands")) {
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

  const std::array<std::uint32_t, 1> v_rcp_f32_words{MakeVop1(42u, 17u, 257u)};
  if (!Expect(decoder.DecodeInstruction(v_rcp_f32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_RCP_F32 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_RCP_F32",
                                     OperandKind::kVgpr, 17u,
                                     OperandKind::kVgpr, 1u),
              "expected decoded V_RCP_F32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_sqrt_f64_words{
      MakeVop1(52u, 18u, 120u)};
  if (!Expect(decoder.DecodeInstruction(v_sqrt_f64_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_SQRT_F64 decode success") ||
      !Expect(ExpectUnaryInstruction(instruction, "V_SQRT_F64",
                                     OperandKind::kVgpr, 18u,
                                     OperandKind::kSgpr, 120u),
              "expected decoded V_SQRT_F64 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kVectorRegister, OperandAccess::kWrite,
                  FragmentKind::kVector, 64u, 2u, false),
              "expected V_SQRT_F64 destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 64u, 2u, false),
              "expected V_SQRT_F64 source descriptor")) {
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

  const std::array<std::uint32_t, 1> vector_add_f16_words{
      MakeVop2(50u, 10u, 257u, 4u)};
  if (!Expect(decoder.DecodeInstruction(vector_add_f16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_ADD_F16 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_ADD_F16",
                                      OperandKind::kVgpr, 10u,
                                      OperandKind::kVgpr, 1u,
                                      OperandKind::kVgpr, 4u),
              "expected decoded V_ADD_F16 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_mul_f16_words{
      MakeVop2(53u, 11u, 258u, 4u)};
  if (!Expect(decoder.DecodeInstruction(vector_mul_f16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_MUL_F16 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_MUL_F16",
                                      OperandKind::kVgpr, 11u,
                                      OperandKind::kVgpr, 2u,
                                      OperandKind::kVgpr, 4u),
              "expected decoded V_MUL_F16 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_subrev_f16_words{
      MakeVop2(52u, 11u, 5u, 6u)};
  if (!Expect(decoder.DecodeInstruction(vector_subrev_f16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_SUBREV_F16 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_SUBREV_F16",
                                      OperandKind::kVgpr, 11u,
                                      OperandKind::kSgpr, 5u,
                                      OperandKind::kVgpr, 6u),
              "expected decoded V_SUBREV_F16 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 2> vector_max_num_f16_literal_words{
      MakeVop2(49u, 12u, 255u, 7u), 0x00003e00u};
  if (!Expect(decoder.DecodeInstruction(vector_max_num_f16_literal_words,
                                        &instruction, &words_consumed,
                                        &error_message),
              "expected V_MAX_NUM_F16 literal decode success") ||
      !Expect(words_consumed == 2u,
              "expected V_MAX_NUM_F16 literal decode to consume 2 dwords") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_MAX_NUM_F16",
                                      OperandKind::kVgpr, 12u,
                                      OperandKind::kImm32, 0x00003e00u,
                                      OperandKind::kVgpr, 7u),
              "expected decoded V_MAX_NUM_F16 literal operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_cvt_pk_rtz_f16_f32_words{
      MakeVop2(47u, 13u, 257u, 2u)};
  if (!Expect(decoder.DecodeInstruction(vector_cvt_pk_rtz_f16_f32_words,
                                        &instruction, &words_consumed,
                                        &error_message),
              "expected V_CVT_PK_RTZ_F16_F32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_CVT_PK_RTZ_F16_F32",
                                      OperandKind::kVgpr, 13u,
                                      OperandKind::kVgpr, 1u,
                                      OperandKind::kVgpr, 2u),
              "expected decoded V_CVT_PK_RTZ_F16_F32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_ldexp_f16_words{
      MakeVop2(59u, 14u, 260u, 5u)};
  if (!Expect(decoder.DecodeInstruction(vector_ldexp_f16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_LDEXP_F16 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_LDEXP_F16",
                                      OperandKind::kVgpr, 14u,
                                      OperandKind::kVgpr, 4u,
                                      OperandKind::kVgpr, 5u),
              "expected decoded V_LDEXP_F16 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_add_f32_words{
      MakeVop2(3u, 15u, 257u, 2u)};
  if (!Expect(decoder.DecodeInstruction(vector_add_f32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_ADD_F32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_ADD_F32",
                                      OperandKind::kVgpr, 15u,
                                      OperandKind::kVgpr, 1u,
                                      OperandKind::kVgpr, 2u),
              "expected decoded V_ADD_F32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_subrev_f32_words{
      MakeVop2(5u, 16u, 257u, 2u)};
  if (!Expect(decoder.DecodeInstruction(vector_subrev_f32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_SUBREV_F32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_SUBREV_F32",
                                      OperandKind::kVgpr, 16u,
                                      OperandKind::kVgpr, 1u,
                                      OperandKind::kVgpr, 2u),
              "expected decoded V_SUBREV_F32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_max_num_f32_words{
      MakeVop2(22u, 17u, 257u, 2u)};
  if (!Expect(decoder.DecodeInstruction(vector_max_num_f32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_MAX_NUM_F32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_MAX_NUM_F32",
                                      OperandKind::kVgpr, 17u,
                                      OperandKind::kVgpr, 1u,
                                      OperandKind::kVgpr, 2u),
              "expected decoded V_MAX_NUM_F32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_add_f64_words{
      MakeVop2(2u, 18u, 257u, 3u)};
  if (!Expect(decoder.DecodeInstruction(vector_add_f64_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_ADD_F64 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_ADD_F64",
                                      OperandKind::kVgpr, 18u,
                                      OperandKind::kVgpr, 1u,
                                      OperandKind::kVgpr, 3u),
              "expected decoded V_ADD_F64 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_max_num_f64_words{
      MakeVop2(14u, 20u, 257u, 3u)};
  if (!Expect(decoder.DecodeInstruction(vector_max_num_f64_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_MAX_NUM_F64 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_MAX_NUM_F64",
                                      OperandKind::kVgpr, 20u,
                                      OperandKind::kVgpr, 1u,
                                      OperandKind::kVgpr, 3u),
              "expected decoded V_MAX_NUM_F64 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_xnor_b32_words{
      MakeVop2(30u, 21u, 261u, 6u)};
  if (!Expect(decoder.DecodeInstruction(vector_xnor_b32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_XNOR_B32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_XNOR_B32",
                                      OperandKind::kVgpr, 21u,
                                      OperandKind::kVgpr, 5u,
                                      OperandKind::kVgpr, 6u),
              "expected decoded V_XNOR_B32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_mul_hi_i32_i24_words{
      MakeVop2(10u, 22u, 261u, 6u)};
  if (!Expect(decoder.DecodeInstruction(vector_mul_hi_i32_i24_words,
                                        &instruction, &words_consumed,
                                        &error_message),
              "expected V_MUL_HI_I32_I24 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_MUL_HI_I32_I24",
                                      OperandKind::kVgpr, 22u,
                                      OperandKind::kVgpr, 5u,
                                      OperandKind::kVgpr, 6u),
              "expected decoded V_MUL_HI_I32_I24 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_lshlrev_b64_words{
      MakeVop2(31u, 24u, 263u, 8u)};
  if (!Expect(decoder.DecodeInstruction(vector_lshlrev_b64_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_LSHLREV_B64 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_LSHLREV_B64",
                                      OperandKind::kVgpr, 24u,
                                      OperandKind::kVgpr, 7u,
                                      OperandKind::kVgpr, 8u),
              "expected decoded V_LSHLREV_B64 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_mul_dx9_zero_f32_words{
      MakeVop2(7u, 25u, 257u, 6u)};
  if (!Expect(decoder.DecodeInstruction(vector_mul_dx9_zero_f32_words,
                                        &instruction, &words_consumed,
                                        &error_message),
              "expected V_MUL_DX9_ZERO_F32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_MUL_DX9_ZERO_F32",
                                      OperandKind::kVgpr, 25u,
                                      OperandKind::kVgpr, 1u,
                                      OperandKind::kVgpr, 6u),
              "expected decoded V_MUL_DX9_ZERO_F32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_fmac_f32_words{
      MakeVop2(43u, 26u, 257u, 6u)};
  if (!Expect(decoder.DecodeInstruction(vector_fmac_f32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_FMAC_F32 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_FMAC_F32",
                                      OperandKind::kVgpr, 26u,
                                      OperandKind::kVgpr, 1u,
                                      OperandKind::kVgpr, 6u),
              "expected decoded V_FMAC_F32 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kVectorRegister,
                  OperandAccess::kReadWrite, FragmentKind::kVector, 32u, 1u,
                  false),
              "expected V_FMAC_F32 destination readwrite descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_fmac_f16_words{
      MakeVop2(54u, 27u, 258u, 7u)};
  if (!Expect(decoder.DecodeInstruction(vector_fmac_f16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_FMAC_F16 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_FMAC_F16",
                                      OperandKind::kVgpr, 27u,
                                      OperandKind::kVgpr, 2u,
                                      OperandKind::kVgpr, 7u),
              "expected decoded V_FMAC_F16 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kVectorRegister,
                  OperandAccess::kReadWrite, FragmentKind::kVector, 16u, 1u,
                  false),
              "expected V_FMAC_F16 destination readwrite descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 2> vector_fmamk_f16_words{
      MakeVop2(55u, 28u, 257u, 7u), 0x00003c00u};
  if (!Expect(decoder.DecodeInstruction(vector_fmamk_f16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_FMAMK_F16 decode success") ||
      !Expect(words_consumed == 2u,
              "expected V_FMAMK_F16 decode to consume 2 dwords") ||
      !Expect(ExpectFourOperandInstruction(
                  instruction, "V_FMAMK_F16", OperandKind::kVgpr, 28u,
                  OperandKind::kVgpr, 1u, OperandKind::kVgpr, 7u,
                  OperandKind::kImm32, 0x00003c00u),
              "expected decoded V_FMAMK_F16 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kVectorRegister, OperandAccess::kWrite,
                  FragmentKind::kVector, 16u, 1u, false),
              "expected V_FMAMK_F16 destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[3], OperandRole::kSource2,
                  OperandSlotKind::kSource2, OperandValueClass::kUnknown,
                  OperandAccess::kRead, FragmentKind::kScalar, 16u, 1u, false),
              "expected V_FMAMK_F16 literal descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 2> vector_fmaak_f16_words{
      MakeVop2(56u, 29u, 255u, 8u), 0x00004000u};
  if (!Expect(decoder.DecodeInstruction(vector_fmaak_f16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_FMAAK_F16 decode success") ||
      !Expect(words_consumed == 2u,
              "expected V_FMAAK_F16 decode to consume 2 dwords") ||
      !Expect(ExpectFourOperandInstruction(
                  instruction, "V_FMAAK_F16", OperandKind::kVgpr, 29u,
                  OperandKind::kImm32, 0x00004000u, OperandKind::kVgpr, 8u,
                  OperandKind::kImm32, 0x00004000u),
              "expected decoded V_FMAAK_F16 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0, OperandValueClass::kUnknown,
                  OperandAccess::kRead, FragmentKind::kScalar, 16u, 1u, false),
              "expected V_FMAAK_F16 source0 shared-literal descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[3], OperandRole::kSource2,
                  OperandSlotKind::kSource2, OperandValueClass::kUnknown,
                  OperandAccess::kRead, FragmentKind::kScalar, 16u, 1u, false),
              "expected V_FMAAK_F16 literal descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_pk_fmac_f16_words{
      MakeVop2(60u, 30u, 257u, 2u)};
  if (!Expect(decoder.DecodeInstruction(vector_pk_fmac_f16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_PK_FMAC_F16 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_PK_FMAC_F16",
                                      OperandKind::kVgpr, 30u,
                                      OperandKind::kVgpr, 1u,
                                      OperandKind::kVgpr, 2u),
              "expected decoded V_PK_FMAC_F16 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kDestination,
                  OperandValueClass::kPackedVector,
                  OperandAccess::kReadWrite, FragmentKind::kPacked, 16u, 2u,
                  false),
              "expected V_PK_FMAC_F16 destination packed readwrite descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kPackedVector, OperandAccess::kRead,
                  FragmentKind::kPacked, 16u, 2u, false),
              "expected V_PK_FMAC_F16 source0 packed descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[2], OperandRole::kSource1,
                  OperandSlotKind::kSource1,
                  OperandValueClass::kPackedVector, OperandAccess::kRead,
                  FragmentKind::kPacked, 16u, 2u, false),
              "expected V_PK_FMAC_F16 source1 packed descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_add_co_ci_u32_words{
      MakeVop2(32u, 29u, 257u, 3u)};
  if (!Expect(decoder.DecodeInstruction(vector_add_co_ci_u32_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_ADD_CO_CI_U32 decode success") ||
      !Expect(instruction.opcode == "V_ADD_CO_CI_U32",
              "expected decoded V_ADD_CO_CI_U32 opcode") ||
      !Expect(instruction.operand_count == 5u,
              "expected five V_ADD_CO_CI_U32 operands") ||
      !Expect(instruction.operands[0].kind == OperandKind::kVgpr &&
                  instruction.operands[0].index == 29u,
              "expected V_ADD_CO_CI_U32 VGPR destination") ||
      !Expect(instruction.operands[1].kind == OperandKind::kSgpr &&
                  instruction.operands[1].index == kImplicitVccPairSgprIndex,
              "expected V_ADD_CO_CI_U32 implicit VCC destination") ||
      !Expect(instruction.operands[2].kind == OperandKind::kVgpr &&
                  instruction.operands[2].index == 1u,
              "expected V_ADD_CO_CI_U32 source0") ||
      !Expect(instruction.operands[3].kind == OperandKind::kVgpr &&
                  instruction.operands[3].index == 3u,
              "expected V_ADD_CO_CI_U32 source1") ||
      !Expect(instruction.operands[4].kind == OperandKind::kSgpr &&
                  instruction.operands[4].index == kImplicitVccPairSgprIndex,
              "expected V_ADD_CO_CI_U32 implicit VCC source") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kDestination,
                  OperandSlotKind::kScalarDestination,
                  OperandValueClass::kScalarRegister, OperandAccess::kWrite,
                  FragmentKind::kScalar, 64u, 2u, true),
              "expected V_ADD_CO_CI_U32 implicit VCC destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[4], OperandRole::kSource2,
                  OperandSlotKind::kSource2,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 64u, 2u, true),
              "expected V_ADD_CO_CI_U32 implicit VCC source descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_subrev_co_ci_u32_words{
      MakeVop2(34u, 31u, 257u, 6u)};
  if (!Expect(decoder.DecodeInstruction(vector_subrev_co_ci_u32_words,
                                        &instruction, &words_consumed,
                                        &error_message),
              "expected V_SUBREV_CO_CI_U32 decode success") ||
      !Expect(instruction.opcode == "V_SUBREV_CO_CI_U32",
              "expected decoded V_SUBREV_CO_CI_U32 opcode") ||
      !Expect(instruction.operand_count == 5u,
              "expected five V_SUBREV_CO_CI_U32 operands")) {
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

  const std::array<std::uint32_t, 1> v_cmp_eq_i16_words{
      MakeVopc(50u, 1u, 4u)};
  if (!Expect(decoder.DecodeInstruction(v_cmp_eq_i16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CMP_EQ_I16 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_CMP_EQ_I16",
                                      OperandKind::kSgpr,
                                      kImplicitVccPairSgprIndex,
                                      OperandKind::kSgpr, 1u,
                                      OperandKind::kVgpr, 4u),
              "expected decoded V_CMP_EQ_I16 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kScalarDestination,
                  OperandValueClass::kScalarRegister, OperandAccess::kWrite,
                  FragmentKind::kScalar, 64u, 2u, true),
              "expected implicit VCC destination descriptor for V_CMP_EQ_I16")
      ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 32u, 1u, false),
              "expected V_CMP_EQ_I16 source0 descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[2], OperandRole::kSource1,
                  OperandSlotKind::kSource1,
                  OperandValueClass::kVectorRegister, OperandAccess::kRead,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_CMP_EQ_I16 source1 descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 2> v_cmpx_eq_u16_literal_words{
      MakeVopc(186u, 255u, 5u), 0x0000ffffu};
  if (!Expect(decoder.DecodeInstruction(v_cmpx_eq_u16_literal_words,
                                        &instruction, &words_consumed,
                                        &error_message),
              "expected V_CMPX_EQ_U16 literal decode success") ||
      !Expect(words_consumed == 2u,
              "expected literal V_CMPX_EQ_U16 decode to consume 2 dwords") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_CMPX_EQ_U16",
                                      OperandKind::kSgpr,
                                      kImplicitVccPairSgprIndex,
                                      OperandKind::kImm32, 0x0000ffffu,
                                      OperandKind::kVgpr, 5u),
              "expected decoded V_CMPX_EQ_U16 literal operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0, OperandValueClass::kUnknown,
                  OperandAccess::kRead, FragmentKind::kScalar, 32u, 1u, false),
              "expected V_CMPX_EQ_U16 literal source descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[2], OperandRole::kSource1,
                  OperandSlotKind::kSource1,
                  OperandValueClass::kVectorRegister, OperandAccess::kRead,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_CMPX_EQ_U16 source1 descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> v_cmp_eq_f16_words{
      MakeVopc(2u, 110u, 70u)};
  if (!Expect(decoder.DecodeInstruction(v_cmp_eq_f16_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_CMP_EQ_F16 decode success") ||
      !Expect(ExpectBinaryInstruction(instruction, "V_CMP_EQ_F16",
                                      OperandKind::kSgpr,
                                      kImplicitVccPairSgprIndex,
                                      OperandKind::kSgpr, 110u,
                                      OperandKind::kVgpr, 70u),
              "expected decoded V_CMP_EQ_F16 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kScalarDestination,
                  OperandValueClass::kScalarRegister, OperandAccess::kWrite,
                  FragmentKind::kScalar, 64u, 2u, true),
              "expected implicit VCC destination descriptor for V_CMP_EQ_F16")
      ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 32u, 1u, false),
              "expected V_CMP_EQ_F16 source0 descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[2], OperandRole::kSource1,
                  OperandSlotKind::kSource1,
                  OperandValueClass::kVectorRegister, OperandAccess::kRead,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_CMP_EQ_F16 source1 descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 2> v_cmpx_class_f16_literal_words{
      MakeVopc(253u, 255u, 71u), kQuietNaNF16Bits};
  if (!Expect(decoder.DecodeInstruction(v_cmpx_class_f16_literal_words,
                                        &instruction, &words_consumed,
                                        &error_message),
              "expected V_CMPX_CLASS_F16 literal decode success") ||
      !Expect(words_consumed == 2u,
              "expected literal V_CMPX_CLASS_F16 decode to consume 2 dwords")
      ||
      !Expect(ExpectBinaryInstruction(instruction, "V_CMPX_CLASS_F16",
                                      OperandKind::kSgpr,
                                      kImplicitVccPairSgprIndex,
                                      OperandKind::kImm32, kQuietNaNF16Bits,
                                      OperandKind::kVgpr, 71u),
              "expected decoded V_CMPX_CLASS_F16 operands") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0, OperandValueClass::kUnknown,
                  OperandAccess::kRead, FragmentKind::kScalar, 32u, 1u, false),
              "expected V_CMPX_CLASS_F16 literal source descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  instruction.operands[2], OperandRole::kSource1,
                  OperandSlotKind::kSource1,
                  OperandValueClass::kVectorRegister, OperandAccess::kRead,
                  FragmentKind::kVector, 32u, 1u, false),
              "expected V_CMPX_CLASS_F16 source1 descriptor")) {
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

  const std::array<std::uint32_t, 21> unary_math_words{
      MakeVop1(1u, 1u, 255u),
      FloatBits(4.0f),
      MakeVop1(1u, 2u, 255u),
      FloatBits(3.0f),
      MakeVop1(1u, 3u, 255u),
      FloatBits(8.0f),
      MakeVop1(1u, 4u, 255u),
      FloatBits(0.0f),
      MakeVop1(42u, 10u, 257u),
      MakeVop1(43u, 11u, 257u),
      MakeVop1(46u, 12u, 257u),
      MakeVop1(51u, 13u, 257u),
      MakeVop1(37u, 14u, 258u),
      MakeVop1(39u, 15u, 259u),
      MakeVop1(53u, 16u, 260u),
      MakeVop1(54u, 17u, 260u),
      MakeVop1(16u, 20u, 257u),
      MakeVop1(47u, 30u, 276u),
      MakeVop1(49u, 32u, 276u),
      MakeVop1(52u, 34u, 276u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> unary_math_program;
  if (!Expect(decoder.DecodeProgram(unary_math_words, &unary_math_program,
                                    &error_message),
              "expected unary math program decode success") ||
      !Expect(unary_math_program.size() == 17u,
              "expected seventeen decoded unary math instructions") ||
      !Expect(unary_math_program[4].opcode == "V_RCP_F32",
              "expected decoded V_RCP_F32") ||
      !Expect(unary_math_program[5].opcode == "V_RCP_IFLAG_F32",
              "expected decoded V_RCP_IFLAG_F32") ||
      !Expect(unary_math_program[6].opcode == "V_RSQ_F32",
              "expected decoded V_RSQ_F32") ||
      !Expect(unary_math_program[7].opcode == "V_SQRT_F32",
              "expected decoded V_SQRT_F32") ||
      !Expect(unary_math_program[8].opcode == "V_EXP_F32",
              "expected decoded V_EXP_F32") ||
      !Expect(unary_math_program[9].opcode == "V_LOG_F32",
              "expected decoded V_LOG_F32") ||
      !Expect(unary_math_program[10].opcode == "V_SIN_F32",
              "expected decoded V_SIN_F32") ||
      !Expect(unary_math_program[11].opcode == "V_COS_F32",
              "expected decoded V_COS_F32") ||
      !Expect(unary_math_program[13].opcode == "V_RCP_F64",
              "expected decoded V_RCP_F64") ||
      !Expect(unary_math_program[14].opcode == "V_RSQ_F64",
              "expected decoded V_RSQ_F64") ||
      !Expect(unary_math_program[15].opcode == "V_SQRT_F64",
              "expected decoded V_SQRT_F64")) {
    return 1;
  }

  auto initialize_unary_math_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;
    state->vgprs[10][2] = 0x10101010u;
    state->vgprs[11][2] = 0x11111111u;
    state->vgprs[12][2] = 0x12121212u;
    state->vgprs[13][2] = 0x13131313u;
    state->vgprs[14][2] = 0x14141414u;
    state->vgprs[15][2] = 0x15151515u;
    state->vgprs[16][2] = 0x16161616u;
    state->vgprs[17][2] = 0x17171717u;
    state->vgprs[20][2] = 0x20202020u;
    state->vgprs[21][2] = 0x21212121u;
    state->vgprs[30][2] = 0x30303030u;
    state->vgprs[31][2] = 0x31313131u;
    state->vgprs[32][2] = 0x32323232u;
    state->vgprs[33][2] = 0x33333333u;
    state->vgprs[34][2] = 0x34343434u;
    state->vgprs[35][2] = 0x35353535u;
  };

  WaveExecutionState decoded_unary_math_state;
  initialize_unary_math_state(&decoded_unary_math_state);
  if (!Expect(interpreter.ExecuteProgram(unary_math_program,
                                         &decoded_unary_math_state,
                                         &error_message),
              "expected decoded unary math execution success") ||
      !Expect(ExpectUnaryMathSeedState(decoded_unary_math_state),
              "expected decoded unary math state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_unary_math_program;
  if (!Expect(interpreter.CompileProgram(unary_math_program,
                                         &compiled_unary_math_program,
                                         &error_message),
              "expected compiled unary math program success") ||
      !Expect(compiled_unary_math_program[4].opcode ==
                  Gfx1201CompiledOpcode::kVRcpF32,
              "expected compiled V_RCP_F32 opcode") ||
      !Expect(compiled_unary_math_program[5].opcode ==
                  Gfx1201CompiledOpcode::kVRcpIflagF32,
              "expected compiled V_RCP_IFLAG_F32 opcode") ||
      !Expect(compiled_unary_math_program[6].opcode ==
                  Gfx1201CompiledOpcode::kVRsqF32,
              "expected compiled V_RSQ_F32 opcode") ||
      !Expect(compiled_unary_math_program[7].opcode ==
                  Gfx1201CompiledOpcode::kVSqrtF32,
              "expected compiled V_SQRT_F32 opcode") ||
      !Expect(compiled_unary_math_program[8].opcode ==
                  Gfx1201CompiledOpcode::kVExpF32,
              "expected compiled V_EXP_F32 opcode") ||
      !Expect(compiled_unary_math_program[9].opcode ==
                  Gfx1201CompiledOpcode::kVLogF32,
              "expected compiled V_LOG_F32 opcode") ||
      !Expect(compiled_unary_math_program[10].opcode ==
                  Gfx1201CompiledOpcode::kVSinF32,
              "expected compiled V_SIN_F32 opcode") ||
      !Expect(compiled_unary_math_program[11].opcode ==
                  Gfx1201CompiledOpcode::kVCosF32,
              "expected compiled V_COS_F32 opcode") ||
      !Expect(compiled_unary_math_program[13].opcode ==
                  Gfx1201CompiledOpcode::kVRcpF64,
              "expected compiled V_RCP_F64 opcode") ||
      !Expect(compiled_unary_math_program[14].opcode ==
                  Gfx1201CompiledOpcode::kVRsqF64,
              "expected compiled V_RSQ_F64 opcode") ||
      !Expect(compiled_unary_math_program[15].opcode ==
                  Gfx1201CompiledOpcode::kVSqrtF64,
              "expected compiled V_SQRT_F64 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_unary_math_state;
  initialize_unary_math_state(&compiled_unary_math_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_unary_math_program,
                                         &compiled_unary_math_state,
                                         &error_message),
              "expected compiled unary math execution success") ||
      !Expect(ExpectUnaryMathSeedState(compiled_unary_math_state),
              "expected compiled unary math state")) {
    return 1;
  }

  const std::array<std::uint32_t, 8> unary_count_convert_words{
      MakeVop1(57u, 10u, 257u),
      MakeVop1(58u, 11u, 257u),
      MakeVop1(59u, 12u, 258u),
      MakeVop1(12u, 13u, 259u),
      MakeVop1(13u, 14u, 260u),
      MakeVop1(106u, 15u, 261u),
      MakeVop1(107u, 16u, 262u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> unary_count_convert_program;
  if (!Expect(decoder.DecodeProgram(unary_count_convert_words,
                                    &unary_count_convert_program,
                                    &error_message),
              "expected unary count/convert program decode success") ||
      !Expect(unary_count_convert_program.size() == 8u,
              "expected eight decoded unary count/convert instructions") ||
      !Expect(unary_count_convert_program[0].opcode == "V_CLZ_I32_U32",
              "expected decoded V_CLZ_I32_U32") ||
      !Expect(unary_count_convert_program[1].opcode == "V_CTZ_I32_B32",
              "expected decoded V_CTZ_I32_B32") ||
      !Expect(unary_count_convert_program[2].opcode == "V_CLS_I32",
              "expected decoded V_CLS_I32") ||
      !Expect(unary_count_convert_program[3].opcode == "V_CVT_NEAREST_I32_F32",
              "expected decoded V_CVT_NEAREST_I32_F32") ||
      !Expect(unary_count_convert_program[4].opcode == "V_CVT_FLOOR_I32_F32",
              "expected decoded V_CVT_FLOOR_I32_F32") ||
      !Expect(unary_count_convert_program[5].opcode == "V_CVT_I32_I16",
              "expected decoded V_CVT_I32_I16") ||
      !Expect(unary_count_convert_program[6].opcode == "V_CVT_U32_U16",
              "expected decoded V_CVT_U32_U16")) {
    return 1;
  }

  auto initialize_unary_count_convert_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;

    state->vgprs[1][0] = 0x0000f000u;
    state->vgprs[1][1] = 0u;
    state->vgprs[1][2] = 0x01010101u;
    state->vgprs[1][3] = 1u;

    state->vgprs[2][0] = 0xffff0000u;
    state->vgprs[2][1] = 0xffffffffu;
    state->vgprs[2][2] = 0x02020202u;
    state->vgprs[2][3] = 0x7fffffffu;

    state->vgprs[3][0] = FloatBits(-1.25f);
    state->vgprs[3][1] = FloatBits(2.5f);
    state->vgprs[3][2] = FloatBits(-7.0f);
    state->vgprs[3][3] = FloatBits(3.6f);

    state->vgprs[4][0] = FloatBits(-1.25f);
    state->vgprs[4][1] = FloatBits(2.5f);
    state->vgprs[4][2] = FloatBits(-8.0f);
    state->vgprs[4][3] = FloatBits(3.6f);

    state->vgprs[5][0] = 0x0000fffdu;
    state->vgprs[5][1] = 0x00000002u;
    state->vgprs[5][2] = 0x05050505u;
    state->vgprs[5][3] = 0x00008001u;

    state->vgprs[6][0] = 0x0000fffdu;
    state->vgprs[6][1] = 0x00000002u;
    state->vgprs[6][2] = 0x06060606u;
    state->vgprs[6][3] = 0x00008001u;

    state->vgprs[10][2] = 0x10101010u;
    state->vgprs[11][2] = 0x11111111u;
    state->vgprs[12][2] = 0x12121212u;
    state->vgprs[13][2] = 0x13131313u;
    state->vgprs[14][2] = 0x14141414u;
    state->vgprs[15][2] = 0x15151515u;
    state->vgprs[16][2] = 0x16161616u;
  };

  WaveExecutionState decoded_unary_count_convert_state;
  initialize_unary_count_convert_state(&decoded_unary_count_convert_state);
  if (!Expect(interpreter.ExecuteProgram(unary_count_convert_program,
                                         &decoded_unary_count_convert_state,
                                         &error_message),
              "expected decoded unary count/convert execution success") ||
      !Expect(ExpectUnaryCountConvertSeedState(
                  decoded_unary_count_convert_state),
              "expected decoded unary count/convert state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_unary_count_convert_program;
  if (!Expect(interpreter.CompileProgram(unary_count_convert_program,
                                         &compiled_unary_count_convert_program,
                                         &error_message),
              "expected compiled unary count/convert program success") ||
      !Expect(compiled_unary_count_convert_program[0].opcode ==
                  Gfx1201CompiledOpcode::kVClzI32U32,
              "expected compiled V_CLZ_I32_U32 opcode") ||
      !Expect(compiled_unary_count_convert_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVCtzI32B32,
              "expected compiled V_CTZ_I32_B32 opcode") ||
      !Expect(compiled_unary_count_convert_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVClsI32,
              "expected compiled V_CLS_I32 opcode") ||
      !Expect(compiled_unary_count_convert_program[3].opcode ==
                  Gfx1201CompiledOpcode::kVCvtNearestI32F32,
              "expected compiled V_CVT_NEAREST_I32_F32 opcode") ||
      !Expect(compiled_unary_count_convert_program[4].opcode ==
                  Gfx1201CompiledOpcode::kVCvtFloorI32F32,
              "expected compiled V_CVT_FLOOR_I32_F32 opcode") ||
      !Expect(compiled_unary_count_convert_program[5].opcode ==
                  Gfx1201CompiledOpcode::kVCvtI32I16,
              "expected compiled V_CVT_I32_I16 opcode") ||
      !Expect(compiled_unary_count_convert_program[6].opcode ==
                  Gfx1201CompiledOpcode::kVCvtU32U16,
              "expected compiled V_CVT_U32_U16 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_unary_count_convert_state;
  initialize_unary_count_convert_state(&compiled_unary_count_convert_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_unary_count_convert_program,
                                         &compiled_unary_count_convert_state,
                                         &error_message),
              "expected compiled unary count/convert execution success") ||
      !Expect(ExpectUnaryCountConvertSeedState(
                  compiled_unary_count_convert_state),
              "expected compiled unary count/convert state")) {
    return 1;
  }

  const std::array<std::uint32_t, 10> f16_bridge_words{
      MakeVop1(0u, 0u, 0u),
      MakeVop1(10u, 40u, 257u),
      MakeVop1(81u, 41u, 258u),
      MakeVop1(80u, 42u, 259u),
      MakeVop1(11u, 43u, 296u),
      MakeVop1(11u, 44u, 297u),
      MakeVop1(11u, 45u, 298u),
      MakeVop1(83u, 46u, 296u),
      MakeVop1(82u, 47u, 298u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> f16_bridge_program;
  if (!Expect(decoder.DecodeProgram(f16_bridge_words, &f16_bridge_program,
                                    &error_message),
              "expected F16 bridge program decode success") ||
      !Expect(f16_bridge_program.size() == 10u,
              "expected ten decoded F16 bridge instructions") ||
      !Expect(f16_bridge_program[0].opcode == "V_NOP",
              "expected decoded V_NOP") ||
      !Expect(f16_bridge_program[1].opcode == "V_CVT_F16_F32",
              "expected decoded V_CVT_F16_F32") ||
      !Expect(f16_bridge_program[2].opcode == "V_CVT_F16_I16",
              "expected decoded V_CVT_F16_I16") ||
      !Expect(f16_bridge_program[3].opcode == "V_CVT_F16_U16",
              "expected decoded V_CVT_F16_U16") ||
      !Expect(f16_bridge_program[4].opcode == "V_CVT_F32_F16",
              "expected decoded V_CVT_F32_F16 from V_CVT_F16_F32 result") ||
      !Expect(f16_bridge_program[5].opcode == "V_CVT_F32_F16",
              "expected decoded V_CVT_F32_F16 from V_CVT_F16_I16 result") ||
      !Expect(f16_bridge_program[6].opcode == "V_CVT_F32_F16",
              "expected decoded V_CVT_F32_F16 from V_CVT_F16_U16 result") ||
      !Expect(f16_bridge_program[7].opcode == "V_CVT_I16_F16",
              "expected decoded V_CVT_I16_F16") ||
      !Expect(f16_bridge_program[8].opcode == "V_CVT_U16_F16",
              "expected decoded V_CVT_U16_F16")) {
    return 1;
  }

  auto initialize_f16_bridge_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;

    state->vgprs[1][0] = FloatBits(1.5f);
    state->vgprs[1][1] = FloatBits(-2.0f);
    state->vgprs[1][2] = 0x01010101u;
    state->vgprs[1][3] = FloatBits(0.5f);

    state->vgprs[2][0] = 0x0000fffeu;
    state->vgprs[2][1] = 0x00000003u;
    state->vgprs[2][2] = 0x02020202u;
    state->vgprs[2][3] = 0x00000000u;

    state->vgprs[3][0] = 0x00000002u;
    state->vgprs[3][1] = 0x00000007u;
    state->vgprs[3][2] = 0x03030303u;
    state->vgprs[3][3] = 0x00000001u;

    state->vgprs[40][2] = 0x40404040u;
    state->vgprs[41][2] = 0x41414141u;
    state->vgprs[42][2] = 0x42424242u;
    state->vgprs[43][2] = 0x43434343u;
    state->vgprs[44][2] = 0x44444444u;
    state->vgprs[45][2] = 0x45454545u;
    state->vgprs[46][2] = 0x46464646u;
    state->vgprs[47][2] = 0x47474747u;
  };

  WaveExecutionState decoded_f16_bridge_state;
  initialize_f16_bridge_state(&decoded_f16_bridge_state);
  if (!Expect(interpreter.ExecuteProgram(f16_bridge_program,
                                         &decoded_f16_bridge_state,
                                         &error_message),
              "expected decoded F16 bridge execution success") ||
      !Expect(ExpectF16BridgeSeedState(decoded_f16_bridge_state),
              "expected decoded F16 bridge state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_f16_bridge_program;
  if (!Expect(interpreter.CompileProgram(f16_bridge_program,
                                         &compiled_f16_bridge_program,
                                         &error_message),
              "expected compiled F16 bridge program success") ||
      !Expect(compiled_f16_bridge_program[0].opcode ==
                  Gfx1201CompiledOpcode::kSNop,
              "expected compiled V_NOP opcode") ||
      !Expect(compiled_f16_bridge_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVCvtF16F32,
              "expected compiled V_CVT_F16_F32 opcode") ||
      !Expect(compiled_f16_bridge_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVCvtF16I16,
              "expected compiled V_CVT_F16_I16 opcode") ||
      !Expect(compiled_f16_bridge_program[3].opcode ==
                  Gfx1201CompiledOpcode::kVCvtF16U16,
              "expected compiled V_CVT_F16_U16 opcode") ||
      !Expect(compiled_f16_bridge_program[4].opcode ==
                  Gfx1201CompiledOpcode::kVCvtF32F16,
              "expected compiled V_CVT_F32_F16 opcode") ||
      !Expect(compiled_f16_bridge_program[5].opcode ==
                  Gfx1201CompiledOpcode::kVCvtF32F16,
              "expected second compiled V_CVT_F32_F16 opcode") ||
      !Expect(compiled_f16_bridge_program[6].opcode ==
                  Gfx1201CompiledOpcode::kVCvtF32F16,
              "expected third compiled V_CVT_F32_F16 opcode") ||
      !Expect(compiled_f16_bridge_program[7].opcode ==
                  Gfx1201CompiledOpcode::kVCvtI16F16,
              "expected compiled V_CVT_I16_F16 opcode") ||
      !Expect(compiled_f16_bridge_program[8].opcode ==
                  Gfx1201CompiledOpcode::kVCvtU16F16,
              "expected compiled V_CVT_U16_F16 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_f16_bridge_state;
  initialize_f16_bridge_state(&compiled_f16_bridge_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_f16_bridge_program,
                                         &compiled_f16_bridge_state,
                                         &error_message),
              "expected compiled F16 bridge execution success") ||
      !Expect(ExpectF16BridgeSeedState(compiled_f16_bridge_state),
              "expected compiled F16 bridge state")) {
    return 1;
  }

  const std::array<std::uint32_t, 4> half_consumer_words{
      MakeVop1(28u, 48u, 257u),
      MakeVop1(105u, 49u, 2u),
      MakeVop1(103u, 50u, 260u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> half_consumer_program;
  if (!Expect(decoder.DecodeProgram(half_consumer_words, &half_consumer_program,
                                    &error_message),
              "expected half consumer program decode success") ||
      !Expect(half_consumer_program.size() == 4u,
              "expected four decoded half consumer instructions") ||
      !Expect(half_consumer_program[0].opcode == "V_MOV_B16",
              "expected decoded V_MOV_B16") ||
      !Expect(half_consumer_program[1].opcode == "V_NOT_B16",
              "expected decoded V_NOT_B16") ||
      !Expect(half_consumer_program[2].opcode == "V_PERMLANE64_B32",
              "expected decoded V_PERMLANE64_B32") ||
      !Expect(half_consumer_program[3].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after half consumer batch")) {
    return 1;
  }

  auto initialize_half_consumer_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;

    state->vgprs[1][0] = 0xaaaa1234u;
    state->vgprs[1][1] = 0xbbbb8001u;
    state->vgprs[1][2] = 0x01010101u;
    state->vgprs[1][3] = 0xcccc0000u;

    state->vgprs[4][0] = 0x11223344u;
    state->vgprs[4][1] = 0x55667788u;
    state->vgprs[4][2] = 0x04040404u;
    state->vgprs[4][3] = 0x99aabbccu;

    state->sgprs[2] = 0xdeadbeefu;

    state->vgprs[48][2] = 0x48484848u;
    state->vgprs[49][2] = 0x49494949u;
    state->vgprs[50][2] = 0x50505050u;
  };

  WaveExecutionState decoded_half_consumer_state;
  initialize_half_consumer_state(&decoded_half_consumer_state);
  if (!Expect(interpreter.ExecuteProgram(half_consumer_program,
                                         &decoded_half_consumer_state,
                                         &error_message),
              "expected decoded half consumer execution success") ||
      !Expect(ExpectHalfConsumerSeedState(decoded_half_consumer_state),
              "expected decoded half consumer state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_half_consumer_program;
  if (!Expect(interpreter.CompileProgram(half_consumer_program,
                                         &compiled_half_consumer_program,
                                         &error_message),
              "expected compiled half consumer program success") ||
      !Expect(compiled_half_consumer_program.size() == 4u,
              "expected four compiled half consumer instructions") ||
      !Expect(compiled_half_consumer_program[0].opcode ==
                  Gfx1201CompiledOpcode::kVMovB16,
              "expected compiled V_MOV_B16 opcode") ||
      !Expect(compiled_half_consumer_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVNotB16,
              "expected compiled V_NOT_B16 opcode") ||
      !Expect(compiled_half_consumer_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVPermlane64B32,
              "expected compiled V_PERMLANE64_B32 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_half_consumer_state;
  initialize_half_consumer_state(&compiled_half_consumer_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_half_consumer_program,
                                         &compiled_half_consumer_state,
                                         &error_message),
              "expected compiled half consumer execution success") ||
      !Expect(ExpectHalfConsumerSeedState(compiled_half_consumer_state),
              "expected compiled half consumer state")) {
    return 1;
  }

  const std::array<std::uint32_t, 3> swap_words{
      MakeVop1(101u, 60u, 317u),
      MakeVop1(102u, 62u, 319u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> swap_program;
  if (!Expect(decoder.DecodeProgram(swap_words, &swap_program, &error_message),
              "expected swap program decode success") ||
      !Expect(swap_program.size() == 3u,
              "expected three decoded swap instructions") ||
      !Expect(swap_program[0].opcode == "V_SWAP_B32",
              "expected decoded V_SWAP_B32") ||
      !Expect(swap_program[1].opcode == "V_SWAP_B16",
              "expected decoded V_SWAP_B16") ||
      !Expect(swap_program[2].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after swap batch")) {
    return 1;
  }

  auto initialize_swap_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;

    state->vgprs[60][0] = 0x11111111u;
    state->vgprs[60][1] = 0x22222222u;
    state->vgprs[60][2] = 0x60606060u;
    state->vgprs[60][3] = 0x33333333u;

    state->vgprs[61][0] = 0xaaaaaaaau;
    state->vgprs[61][1] = 0xbbbbbbbbu;
    state->vgprs[61][2] = 0x61616161u;
    state->vgprs[61][3] = 0xccccccccu;

    state->vgprs[62][0] = 0xaaaa1234u;
    state->vgprs[62][1] = 0xbbbb8001u;
    state->vgprs[62][2] = 0x62626262u;
    state->vgprs[62][3] = 0xccccffffu;

    state->vgprs[63][0] = 0xddddbeefu;
    state->vgprs[63][1] = 0xeeee0002u;
    state->vgprs[63][2] = 0x63636363u;
    state->vgprs[63][3] = 0xffff1234u;
  };

  WaveExecutionState decoded_swap_state;
  initialize_swap_state(&decoded_swap_state);
  if (!Expect(interpreter.ExecuteProgram(swap_program, &decoded_swap_state,
                                         &error_message),
              "expected decoded swap execution success") ||
      !Expect(ExpectSwapSeedState(decoded_swap_state),
              "expected decoded swap state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_swap_program;
  if (!Expect(interpreter.CompileProgram(swap_program, &compiled_swap_program,
                                         &error_message),
              "expected compiled swap program success") ||
      !Expect(compiled_swap_program.size() == 3u,
              "expected three compiled swap instructions") ||
      !Expect(compiled_swap_program[0].opcode ==
                  Gfx1201CompiledOpcode::kVSwapB32,
              "expected compiled V_SWAP_B32 opcode") ||
      !Expect(compiled_swap_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVSwapB16,
              "expected compiled V_SWAP_B16 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_swap_state;
  initialize_swap_state(&compiled_swap_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_swap_program,
                                         &compiled_swap_state, &error_message),
              "expected compiled swap execution success") ||
      !Expect(ExpectSwapSeedState(compiled_swap_state),
              "expected compiled swap state")) {
    return 1;
  }

  const std::array<std::uint32_t, 5> offset_norm_words{
      MakeVop1(27u, 0u, 0u),
      MakeVop1(14u, 70u, 266u),
      MakeVop1(99u, 71u, 267u),
      MakeVop1(100u, 72u, 268u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> offset_norm_program;
  if (!Expect(decoder.DecodeProgram(offset_norm_words, &offset_norm_program,
                                    &error_message),
              "expected offset/norm program decode success") ||
      !Expect(offset_norm_program.size() == 5u,
              "expected five decoded offset/norm instructions") ||
      !Expect(offset_norm_program[0].opcode == "V_PIPEFLUSH",
              "expected decoded V_PIPEFLUSH") ||
      !Expect(offset_norm_program[1].opcode == "V_CVT_OFF_F32_I4",
              "expected decoded V_CVT_OFF_F32_I4") ||
      !Expect(offset_norm_program[2].opcode == "V_CVT_NORM_I16_F16",
              "expected decoded V_CVT_NORM_I16_F16") ||
      !Expect(offset_norm_program[3].opcode == "V_CVT_NORM_U16_F16",
              "expected decoded V_CVT_NORM_U16_F16") ||
      !Expect(offset_norm_program[4].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after offset/norm batch")) {
    return 1;
  }

  auto initialize_offset_norm_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;

    state->vgprs[10][0] = 0x00000008u;
    state->vgprs[10][1] = 0x00000002u;
    state->vgprs[10][2] = 0x10101010u;
    state->vgprs[10][3] = 0x0000000fu;

    state->vgprs[11][0] = 0x00003800u;
    state->vgprs[11][1] = 0x0000b800u;
    state->vgprs[11][2] = 0x11111111u;
    state->vgprs[11][3] = 0x00003400u;

    state->vgprs[12][0] = 0x00003400u;
    state->vgprs[12][1] = 0x00003e00u;
    state->vgprs[12][2] = 0x12121212u;
    state->vgprs[12][3] = 0x0000b800u;

    state->vgprs[70][2] = 0x70707070u;
    state->vgprs[71][2] = 0x71717171u;
    state->vgprs[72][2] = 0x72727272u;
  };

  WaveExecutionState decoded_offset_norm_state;
  initialize_offset_norm_state(&decoded_offset_norm_state);
  if (!Expect(interpreter.ExecuteProgram(offset_norm_program,
                                         &decoded_offset_norm_state,
                                         &error_message),
              "expected decoded offset/norm execution success") ||
      !Expect(ExpectOffsetNormSeedState(decoded_offset_norm_state),
              "expected decoded offset/norm state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_offset_norm_program;
  if (!Expect(interpreter.CompileProgram(offset_norm_program,
                                         &compiled_offset_norm_program,
                                         &error_message),
              "expected compiled offset/norm program success") ||
      !Expect(compiled_offset_norm_program.size() == 5u,
              "expected five compiled offset/norm instructions") ||
      !Expect(compiled_offset_norm_program[0].opcode ==
                  Gfx1201CompiledOpcode::kSNop,
              "expected compiled V_PIPEFLUSH opcode") ||
      !Expect(compiled_offset_norm_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVCvtOffF32I4,
              "expected compiled V_CVT_OFF_F32_I4 opcode") ||
      !Expect(compiled_offset_norm_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVCvtNormI16F16,
              "expected compiled V_CVT_NORM_I16_F16 opcode") ||
      !Expect(compiled_offset_norm_program[3].opcode ==
                  Gfx1201CompiledOpcode::kVCvtNormU16F16,
              "expected compiled V_CVT_NORM_U16_F16 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_offset_norm_state;
  initialize_offset_norm_state(&compiled_offset_norm_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_offset_norm_program,
                                         &compiled_offset_norm_state,
                                         &error_message),
              "expected compiled offset/norm execution success") ||
      !Expect(ExpectOffsetNormSeedState(compiled_offset_norm_state),
              "expected compiled offset/norm state")) {
    return 1;
  }

  const std::array<std::uint32_t, 15> f16_unary_words{
      MakeVop1(84u, 80u, 257u),
      MakeVop1(86u, 81u, 257u),
      MakeVop1(85u, 82u, 257u),
      MakeVop1(88u, 83u, 262u),
      MakeVop1(87u, 84u, 258u),
      MakeVop1(96u, 85u, 259u),
      MakeVop1(97u, 86u, 259u),
      MakeVop1(90u, 87u, 257u),
      MakeVop1(89u, 88u, 257u),
      MakeVop1(95u, 89u, 260u),
      MakeVop1(93u, 90u, 261u),
      MakeVop1(92u, 91u, 261u),
      MakeVop1(94u, 92u, 261u),
      MakeVop1(91u, 93u, 261u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> f16_unary_program;
  if (!Expect(decoder.DecodeProgram(f16_unary_words, &f16_unary_program,
                                    &error_message),
              "expected F16 unary program decode success") ||
      !Expect(f16_unary_program.size() == 15u,
              "expected fifteen decoded F16 unary instructions") ||
      !Expect(f16_unary_program[0].opcode == "V_RCP_F16",
              "expected decoded V_RCP_F16") ||
      !Expect(f16_unary_program[1].opcode == "V_RSQ_F16",
              "expected decoded V_RSQ_F16") ||
      !Expect(f16_unary_program[2].opcode == "V_SQRT_F16",
              "expected decoded V_SQRT_F16") ||
      !Expect(f16_unary_program[3].opcode == "V_EXP_F16",
              "expected decoded V_EXP_F16") ||
      !Expect(f16_unary_program[4].opcode == "V_LOG_F16",
              "expected decoded V_LOG_F16") ||
      !Expect(f16_unary_program[5].opcode == "V_SIN_F16",
              "expected decoded V_SIN_F16") ||
      !Expect(f16_unary_program[6].opcode == "V_COS_F16",
              "expected decoded V_COS_F16") ||
      !Expect(f16_unary_program[7].opcode == "V_FREXP_EXP_I16_F16",
              "expected decoded V_FREXP_EXP_I16_F16") ||
      !Expect(f16_unary_program[8].opcode == "V_FREXP_MANT_F16",
              "expected decoded V_FREXP_MANT_F16") ||
      !Expect(f16_unary_program[9].opcode == "V_FRACT_F16",
              "expected decoded V_FRACT_F16") ||
      !Expect(f16_unary_program[10].opcode == "V_TRUNC_F16",
              "expected decoded V_TRUNC_F16") ||
      !Expect(f16_unary_program[11].opcode == "V_CEIL_F16",
              "expected decoded V_CEIL_F16") ||
      !Expect(f16_unary_program[12].opcode == "V_RNDNE_F16",
              "expected decoded V_RNDNE_F16") ||
      !Expect(f16_unary_program[13].opcode == "V_FLOOR_F16",
              "expected decoded V_FLOOR_F16") ||
      !Expect(f16_unary_program[14].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after F16 unary batch")) {
    return 1;
  }

  auto initialize_f16_unary_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;

    state->vgprs[1][0] = 0x00004400u;
    state->vgprs[1][1] = 0x00004400u;
    state->vgprs[1][2] = 0x11111111u;
    state->vgprs[1][3] = 0x00004400u;

    state->vgprs[2][0] = 0x00004800u;
    state->vgprs[2][1] = 0x00004800u;
    state->vgprs[2][2] = 0x22222222u;
    state->vgprs[2][3] = 0x00004800u;

    state->vgprs[3][0] = 0x00000000u;
    state->vgprs[3][1] = 0x00000000u;
    state->vgprs[3][2] = 0x33333333u;
    state->vgprs[3][3] = 0x00000000u;

    state->vgprs[4][0] = 0x00003d00u;
    state->vgprs[4][1] = 0x00003d00u;
    state->vgprs[4][2] = 0x44444444u;
    state->vgprs[4][3] = 0x00003d00u;

    state->vgprs[5][0] = 0x0000c180u;
    state->vgprs[5][1] = 0x0000c180u;
    state->vgprs[5][2] = 0x55555555u;
    state->vgprs[5][3] = 0x0000c180u;

    state->vgprs[6][0] = 0x00004200u;
    state->vgprs[6][1] = 0x00004200u;
    state->vgprs[6][2] = 0x66666666u;
    state->vgprs[6][3] = 0x00004200u;

    state->vgprs[80][2] = 0x80808080u;
    state->vgprs[81][2] = 0x81818181u;
    state->vgprs[82][2] = 0x82828282u;
    state->vgprs[83][2] = 0x83838383u;
    state->vgprs[84][2] = 0x84848484u;
    state->vgprs[85][2] = 0x85858585u;
    state->vgprs[86][2] = 0x86868686u;
    state->vgprs[87][2] = 0x87878787u;
    state->vgprs[88][2] = 0x88888888u;
    state->vgprs[89][2] = 0x89898989u;
    state->vgprs[90][2] = 0x90909090u;
    state->vgprs[91][2] = 0x91919191u;
    state->vgprs[92][2] = 0x92929292u;
    state->vgprs[93][2] = 0x93939393u;
  };

  WaveExecutionState decoded_f16_unary_state;
  initialize_f16_unary_state(&decoded_f16_unary_state);
  if (!Expect(interpreter.ExecuteProgram(f16_unary_program,
                                         &decoded_f16_unary_state,
                                         &error_message),
              "expected decoded F16 unary execution success") ||
      !Expect(ExpectF16UnaryMathSeedState(decoded_f16_unary_state),
              "expected decoded F16 unary state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_f16_unary_program;
  if (!Expect(interpreter.CompileProgram(f16_unary_program,
                                         &compiled_f16_unary_program,
                                         &error_message),
              "expected compiled F16 unary program success") ||
      !Expect(compiled_f16_unary_program.size() == 15u,
              "expected fifteen compiled F16 unary instructions") ||
      !Expect(compiled_f16_unary_program[0].opcode ==
                  Gfx1201CompiledOpcode::kVRcpF16,
              "expected compiled V_RCP_F16 opcode") ||
      !Expect(compiled_f16_unary_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVRsqF16,
              "expected compiled V_RSQ_F16 opcode") ||
      !Expect(compiled_f16_unary_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVSqrtF16,
              "expected compiled V_SQRT_F16 opcode") ||
      !Expect(compiled_f16_unary_program[3].opcode ==
                  Gfx1201CompiledOpcode::kVExpF16,
              "expected compiled V_EXP_F16 opcode") ||
      !Expect(compiled_f16_unary_program[4].opcode ==
                  Gfx1201CompiledOpcode::kVLogF16,
              "expected compiled V_LOG_F16 opcode") ||
      !Expect(compiled_f16_unary_program[5].opcode ==
                  Gfx1201CompiledOpcode::kVSinF16,
              "expected compiled V_SIN_F16 opcode") ||
      !Expect(compiled_f16_unary_program[6].opcode ==
                  Gfx1201CompiledOpcode::kVCosF16,
              "expected compiled V_COS_F16 opcode") ||
      !Expect(compiled_f16_unary_program[7].opcode ==
                  Gfx1201CompiledOpcode::kVFrexpExpI16F16,
              "expected compiled V_FREXP_EXP_I16_F16 opcode") ||
      !Expect(compiled_f16_unary_program[8].opcode ==
                  Gfx1201CompiledOpcode::kVFrexpMantF16,
              "expected compiled V_FREXP_MANT_F16 opcode") ||
      !Expect(compiled_f16_unary_program[9].opcode ==
                  Gfx1201CompiledOpcode::kVFractF16,
              "expected compiled V_FRACT_F16 opcode") ||
      !Expect(compiled_f16_unary_program[10].opcode ==
                  Gfx1201CompiledOpcode::kVTruncF16,
              "expected compiled V_TRUNC_F16 opcode") ||
      !Expect(compiled_f16_unary_program[11].opcode ==
                  Gfx1201CompiledOpcode::kVCeilF16,
              "expected compiled V_CEIL_F16 opcode") ||
      !Expect(compiled_f16_unary_program[12].opcode ==
                  Gfx1201CompiledOpcode::kVRndneF16,
              "expected compiled V_RNDNE_F16 opcode") ||
      !Expect(compiled_f16_unary_program[13].opcode ==
                  Gfx1201CompiledOpcode::kVFloorF16,
              "expected compiled V_FLOOR_F16 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_f16_unary_state;
  initialize_f16_unary_state(&compiled_f16_unary_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_f16_unary_program,
                                         &compiled_f16_unary_state,
                                         &error_message),
              "expected compiled F16 unary execution success") ||
      !Expect(ExpectF16UnaryMathSeedState(compiled_f16_unary_state),
              "expected compiled F16 unary state")) {
    return 1;
  }

  const std::array<std::uint32_t, 2> readfirstlane_words{
      MakeVop1(2u, 60u, 278u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> readfirstlane_program;
  if (!Expect(decoder.DecodeProgram(readfirstlane_words, &readfirstlane_program,
                                    &error_message),
              "expected readfirstlane program decode success") ||
      !Expect(readfirstlane_program.size() == 2u,
              "expected two decoded readfirstlane instructions") ||
      !Expect(readfirstlane_program[0].opcode == "V_READFIRSTLANE_B32",
              "expected decoded V_READFIRSTLANE_B32") ||
      !Expect(readfirstlane_program[1].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after V_READFIRSTLANE_B32")) {
    return 1;
  }

  auto initialize_readfirstlane_state = [](WaveExecutionState* state) {
    state->exec_mask = (1ULL << 31) | (1ULL << 40);
    state->vgprs[22][31] = 0xfeedbeefu;
    state->vgprs[22][5] = 0x11111111u;
    state->sgprs[60] = 0xaaaaaaaau;
  };

  WaveExecutionState decoded_readfirstlane_state;
  initialize_readfirstlane_state(&decoded_readfirstlane_state);
  if (!Expect(interpreter.ExecuteProgram(readfirstlane_program,
                                         &decoded_readfirstlane_state,
                                         &error_message),
              "expected decoded readfirstlane execution success") ||
      !Expect(ExpectReadfirstlaneSeedState(decoded_readfirstlane_state),
              "expected decoded readfirstlane wave32 state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_readfirstlane_program;
  if (!Expect(interpreter.CompileProgram(readfirstlane_program,
                                         &compiled_readfirstlane_program,
                                         &error_message),
              "expected compiled readfirstlane program success") ||
      !Expect(compiled_readfirstlane_program.size() == 2u,
              "expected two compiled readfirstlane instructions") ||
      !Expect(compiled_readfirstlane_program[0].opcode ==
                  Gfx1201CompiledOpcode::kVReadfirstlaneB32,
              "expected compiled V_READFIRSTLANE_B32 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_readfirstlane_state;
  initialize_readfirstlane_state(&compiled_readfirstlane_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_readfirstlane_program,
                                         &compiled_readfirstlane_state,
                                         &error_message),
              "expected compiled readfirstlane execution success") ||
      !Expect(ExpectReadfirstlaneSeedState(compiled_readfirstlane_state),
              "expected compiled readfirstlane wave32 state")) {
    return 1;
  }

  const std::array<std::uint32_t, 6> movrel_words{
      MakeVop1(66u, 10u, 5u),
      MakeVop1(67u, 12u, 276u),
      MakeVop1(68u, 30u, 296u),
      MakeVop1(72u, 50u, 316u),
      MakeVop1(104u, 70u, 336u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> movrel_program;
  if (!Expect(decoder.DecodeProgram(movrel_words, &movrel_program, &error_message),
              "expected movrel program decode success") ||
      !Expect(movrel_program.size() == 6u,
              "expected six decoded movrel instructions") ||
      !Expect(movrel_program[0].opcode == "V_MOVRELD_B32",
              "expected decoded V_MOVRELD_B32") ||
      !Expect(movrel_program[1].opcode == "V_MOVRELS_B32",
              "expected decoded V_MOVRELS_B32") ||
      !Expect(movrel_program[2].opcode == "V_MOVRELSD_B32",
              "expected decoded V_MOVRELSD_B32") ||
      !Expect(movrel_program[3].opcode == "V_MOVRELSD_2_B32",
              "expected decoded V_MOVRELSD_2_B32") ||
      !Expect(movrel_program[4].opcode == "V_SWAPREL_B32",
              "expected decoded V_SWAPREL_B32") ||
      !Expect(movrel_program[5].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after movrel batch")) {
    return 1;
  }

  auto initialize_movrel_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;
    state->sgprs[5] = 0xcafef00du;
    state->sgprs[kM0RegisterIndex] = 0x00030001u;

    state->vgprs[10][0] = 0x10101010u;
    state->vgprs[10][1] = 0x20202020u;
    state->vgprs[10][2] = 0x30303030u;
    state->vgprs[10][3] = 0x40404040u;

    state->vgprs[11][0] = 0x11111111u;
    state->vgprs[11][1] = 0x11111111u;
    state->vgprs[11][2] = 0x11111111u;
    state->vgprs[11][3] = 0x11111111u;

    state->vgprs[12][0] = 0x12121212u;
    state->vgprs[12][1] = 0x12121212u;
    state->vgprs[12][2] = 0x12121212u;
    state->vgprs[12][3] = 0x12121212u;

    state->vgprs[20][0] = 0x20200000u;
    state->vgprs[20][1] = 0x20200001u;
    state->vgprs[20][2] = 0x20200002u;
    state->vgprs[20][3] = 0x20200003u;

    state->vgprs[21][0] = 0xaaaa0001u;
    state->vgprs[21][1] = 0xbbbb0002u;
    state->vgprs[21][2] = 0x41414141u;
    state->vgprs[21][3] = 0xdddd0004u;

    state->vgprs[30][0] = 0x30300000u;
    state->vgprs[30][1] = 0x30300001u;
    state->vgprs[30][2] = 0x30300002u;
    state->vgprs[30][3] = 0x30300003u;

    state->vgprs[31][0] = 0x31313131u;
    state->vgprs[31][1] = 0x31313131u;
    state->vgprs[31][2] = 0x31313131u;
    state->vgprs[31][3] = 0x31313131u;

    state->vgprs[41][0] = 0x12340001u;
    state->vgprs[41][1] = 0x12340002u;
    state->vgprs[41][2] = 0x41414141u;
    state->vgprs[41][3] = 0x12340004u;

    state->vgprs[53][0] = 0x53535353u;
    state->vgprs[53][1] = 0x53535353u;
    state->vgprs[53][2] = 0x53535353u;
    state->vgprs[53][3] = 0x53535353u;

    state->vgprs[61][0] = 0x98760001u;
    state->vgprs[61][1] = 0x98760002u;
    state->vgprs[61][2] = 0x61616161u;
    state->vgprs[61][3] = 0x98760004u;

    state->vgprs[73][0] = 0x11110001u;
    state->vgprs[73][1] = 0x11110002u;
    state->vgprs[73][2] = 0x73737373u;
    state->vgprs[73][3] = 0x11110004u;

    state->vgprs[81][0] = 0x22220001u;
    state->vgprs[81][1] = 0x22220002u;
    state->vgprs[81][2] = 0x81818181u;
    state->vgprs[81][3] = 0x22220004u;
  };

  WaveExecutionState decoded_movrel_state;
  initialize_movrel_state(&decoded_movrel_state);
  if (!Expect(interpreter.ExecuteProgram(movrel_program, &decoded_movrel_state,
                                         &error_message),
              "expected decoded movrel execution success") ||
      !Expect(ExpectMovrelSeedState(decoded_movrel_state),
              "expected decoded movrel state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_movrel_program;
  if (!Expect(interpreter.CompileProgram(movrel_program, &compiled_movrel_program,
                                         &error_message),
              "expected compiled movrel program success") ||
      !Expect(compiled_movrel_program.size() == 6u,
              "expected six compiled movrel instructions") ||
      !Expect(compiled_movrel_program[0].opcode ==
                  Gfx1201CompiledOpcode::kVMovreldB32,
              "expected compiled V_MOVRELD_B32 opcode") ||
      !Expect(compiled_movrel_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVMovrelsB32,
              "expected compiled V_MOVRELS_B32 opcode") ||
      !Expect(compiled_movrel_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVMovrelsdB32,
              "expected compiled V_MOVRELSD_B32 opcode") ||
      !Expect(compiled_movrel_program[3].opcode ==
                  Gfx1201CompiledOpcode::kVMovrelsd2B32,
              "expected compiled V_MOVRELSD_2_B32 opcode") ||
      !Expect(compiled_movrel_program[4].opcode ==
                  Gfx1201CompiledOpcode::kVSwaprelB32,
              "expected compiled V_SWAPREL_B32 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_movrel_state;
  initialize_movrel_state(&compiled_movrel_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_movrel_program,
                                         &compiled_movrel_state,
                                         &error_message),
              "expected compiled movrel execution success") ||
      !Expect(ExpectMovrelSeedState(compiled_movrel_state),
              "expected compiled movrel state")) {
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

  const std::array<std::uint32_t, 6> f16_vector_binary_words{
      MakeVop2(50u, 94u, 257u, 2u),
      MakeVop2(51u, 95u, 257u, 2u),
      MakeVop2(52u, 96u, 257u, 2u),
      MakeVop2(48u, 97u, 257u, 2u),
      MakeVop2(49u, 98u, 257u, 2u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> f16_vector_binary_program;
  if (!Expect(decoder.DecodeProgram(f16_vector_binary_words,
                                    &f16_vector_binary_program, &error_message),
              "expected F16 vector binary program decode success") ||
      !Expect(f16_vector_binary_program.size() == 6u,
              "expected six decoded F16 vector binary instructions") ||
      !Expect(f16_vector_binary_program[0].opcode == "V_ADD_F16",
              "expected decoded V_ADD_F16") ||
      !Expect(f16_vector_binary_program[1].opcode == "V_SUB_F16",
              "expected decoded V_SUB_F16") ||
      !Expect(f16_vector_binary_program[2].opcode == "V_SUBREV_F16",
              "expected decoded V_SUBREV_F16") ||
      !Expect(f16_vector_binary_program[3].opcode == "V_MIN_NUM_F16",
              "expected decoded V_MIN_NUM_F16") ||
      !Expect(f16_vector_binary_program[4].opcode == "V_MAX_NUM_F16",
              "expected decoded V_MAX_NUM_F16") ||
      !Expect(f16_vector_binary_program[5].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after F16 vector binary batch")) {
    return 1;
  }

  auto initialize_f16_vector_binary_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;

    state->vgprs[1][0] = 0x00003e00u;
    state->vgprs[1][1] = 0x00003e00u;
    state->vgprs[1][2] = 0x11111111u;
    state->vgprs[1][3] = 0x00003e00u;

    state->vgprs[2][0] = 0x00003800u;
    state->vgprs[2][1] = 0x00003800u;
    state->vgprs[2][2] = 0x22222222u;
    state->vgprs[2][3] = 0x00003800u;

    state->vgprs[94][2] = 0x94949494u;
    state->vgprs[95][2] = 0x95959595u;
    state->vgprs[96][2] = 0x96969696u;
    state->vgprs[97][2] = 0x97979797u;
    state->vgprs[98][2] = 0x98989898u;
  };

  WaveExecutionState decoded_f16_vector_binary_state;
  initialize_f16_vector_binary_state(&decoded_f16_vector_binary_state);
  if (!Expect(interpreter.ExecuteProgram(f16_vector_binary_program,
                                         &decoded_f16_vector_binary_state,
                                         &error_message),
              "expected decoded F16 vector binary execution success") ||
      !Expect(ExpectF16VectorBinarySeedState(decoded_f16_vector_binary_state),
              "expected decoded F16 vector binary state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_f16_vector_binary_program;
  if (!Expect(interpreter.CompileProgram(f16_vector_binary_program,
                                         &compiled_f16_vector_binary_program,
                                         &error_message),
              "expected compiled F16 vector binary program success") ||
      !Expect(compiled_f16_vector_binary_program.size() == 6u,
              "expected six compiled F16 vector binary instructions") ||
      !Expect(compiled_f16_vector_binary_program[0].opcode ==
                  Gfx1201CompiledOpcode::kVAddF16,
              "expected compiled V_ADD_F16 opcode") ||
      !Expect(compiled_f16_vector_binary_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVSubF16,
              "expected compiled V_SUB_F16 opcode") ||
      !Expect(compiled_f16_vector_binary_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVSubrevF16,
              "expected compiled V_SUBREV_F16 opcode") ||
      !Expect(compiled_f16_vector_binary_program[3].opcode ==
                  Gfx1201CompiledOpcode::kVMinNumF16,
              "expected compiled V_MIN_NUM_F16 opcode") ||
      !Expect(compiled_f16_vector_binary_program[4].opcode ==
                  Gfx1201CompiledOpcode::kVMaxNumF16,
              "expected compiled V_MAX_NUM_F16 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_f16_vector_binary_state;
  initialize_f16_vector_binary_state(&compiled_f16_vector_binary_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_f16_vector_binary_program,
                                         &compiled_f16_vector_binary_state,
                                         &error_message),
              "expected compiled F16 vector binary execution success") ||
      !Expect(ExpectF16VectorBinarySeedState(compiled_f16_vector_binary_state),
              "expected compiled F16 vector binary state")) {
    return 1;
  }

  const std::array<std::uint32_t, 3> half_pack_mul_words{
      MakeVop1(98u, 99u, 257u),
      MakeVop2(53u, 100u, 258u, 3u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> half_pack_mul_program;
  if (!Expect(decoder.DecodeProgram(half_pack_mul_words, &half_pack_mul_program,
                                    &error_message),
              "expected half pack/mul program decode success") ||
      !Expect(half_pack_mul_program.size() == 3u,
              "expected three decoded half pack/mul instructions") ||
      !Expect(half_pack_mul_program[0].opcode == "V_SAT_PK_U8_I16",
              "expected decoded V_SAT_PK_U8_I16") ||
      !Expect(half_pack_mul_program[1].opcode == "V_MUL_F16",
              "expected decoded V_MUL_F16") ||
      !Expect(half_pack_mul_program[2].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after half pack/mul batch")) {
    return 1;
  }

  auto initialize_half_pack_mul_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;

    state->vgprs[1][0] = 0x012cff9cu;
    state->vgprs[1][1] = 0x00640005u;
    state->vgprs[1][2] = 0x11111111u;
    state->vgprs[1][3] = 0xffec00ffu;

    state->vgprs[2][0] = 0x00003e00u;
    state->vgprs[2][1] = 0x00004000u;
    state->vgprs[2][2] = 0x22222222u;
    state->vgprs[2][3] = 0x0000b800u;

    state->vgprs[3][0] = 0x00004000u;
    state->vgprs[3][1] = 0x00003c00u;
    state->vgprs[3][2] = 0x33333333u;
    state->vgprs[3][3] = 0x00004200u;

    state->vgprs[99][2] = 0x99999999u;
    state->vgprs[100][2] = 0xa0a0a0a0u;
  };

  WaveExecutionState decoded_half_pack_mul_state;
  initialize_half_pack_mul_state(&decoded_half_pack_mul_state);
  if (!Expect(interpreter.ExecuteProgram(half_pack_mul_program,
                                         &decoded_half_pack_mul_state,
                                         &error_message),
              "expected decoded half pack/mul execution success") ||
      !Expect(ExpectHalfPackMulSeedState(decoded_half_pack_mul_state),
              "expected decoded half pack/mul state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_half_pack_mul_program;
  if (!Expect(interpreter.CompileProgram(half_pack_mul_program,
                                         &compiled_half_pack_mul_program,
                                         &error_message),
              "expected compiled half pack/mul program success") ||
      !Expect(compiled_half_pack_mul_program.size() == 3u,
              "expected three compiled half pack/mul instructions") ||
      !Expect(compiled_half_pack_mul_program[0].opcode ==
                  Gfx1201CompiledOpcode::kVSatPkU8I16,
              "expected compiled V_SAT_PK_U8_I16 opcode") ||
      !Expect(compiled_half_pack_mul_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVMulF16,
              "expected compiled V_MUL_F16 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_half_pack_mul_state;
  initialize_half_pack_mul_state(&compiled_half_pack_mul_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_half_pack_mul_program,
                                         &compiled_half_pack_mul_state,
                                         &error_message),
              "expected compiled half pack/mul execution success") ||
      !Expect(ExpectHalfPackMulSeedState(compiled_half_pack_mul_state),
              "expected compiled half pack/mul state")) {
    return 1;
  }

  const std::array<std::uint32_t, 3> half_pack_exponent_words{
      MakeVop2(47u, 101u, 257u, 2u),
      MakeVop2(59u, 102u, 260u, 5u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> half_pack_exponent_program;
  if (!Expect(decoder.DecodeProgram(half_pack_exponent_words,
                                    &half_pack_exponent_program,
                                    &error_message),
              "expected half pack/exponent program decode success") ||
      !Expect(half_pack_exponent_program.size() == 3u,
              "expected three decoded half pack/exponent instructions") ||
      !Expect(half_pack_exponent_program[0].opcode == "V_CVT_PK_RTZ_F16_F32",
              "expected decoded V_CVT_PK_RTZ_F16_F32") ||
      !Expect(half_pack_exponent_program[1].opcode == "V_LDEXP_F16",
              "expected decoded V_LDEXP_F16") ||
      !Expect(half_pack_exponent_program[2].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after half pack/exponent batch")) {
    return 1;
  }

  auto initialize_half_pack_exponent_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;

    state->vgprs[1][0] = FloatBits(1.0007f);
    state->vgprs[1][1] = FloatBits(2.5f);
    state->vgprs[1][2] = 0x11111111u;
    state->vgprs[1][3] = FloatBits(-0.75f);

    state->vgprs[2][0] = FloatBits(-1.0007f);
    state->vgprs[2][1] = FloatBits(0.25f);
    state->vgprs[2][2] = 0x22222222u;
    state->vgprs[2][3] = FloatBits(1.5f);

    state->vgprs[4][0] = 0x00003800u;
    state->vgprs[4][1] = 0x0000b800u;
    state->vgprs[4][2] = 0x44444444u;
    state->vgprs[4][3] = 0x00003e00u;

    state->vgprs[5][0] = 0x00000002u;
    state->vgprs[5][1] = 0x0000ffffu;
    state->vgprs[5][2] = 0x55555555u;
    state->vgprs[5][3] = 0x00000001u;

    state->vgprs[101][2] = 0xa1a1a1a1u;
    state->vgprs[102][2] = 0xa2a2a2a2u;
  };

  WaveExecutionState decoded_half_pack_exponent_state;
  initialize_half_pack_exponent_state(&decoded_half_pack_exponent_state);
  if (!Expect(interpreter.ExecuteProgram(half_pack_exponent_program,
                                         &decoded_half_pack_exponent_state,
                                         &error_message),
              "expected decoded half pack/exponent execution success") ||
      !Expect(ExpectHalfPackExponentSeedState(
                  decoded_half_pack_exponent_state),
              "expected decoded half pack/exponent state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_half_pack_exponent_program;
  if (!Expect(interpreter.CompileProgram(half_pack_exponent_program,
                                         &compiled_half_pack_exponent_program,
                                         &error_message),
              "expected compiled half pack/exponent program success") ||
      !Expect(compiled_half_pack_exponent_program.size() == 3u,
              "expected three compiled half pack/exponent instructions") ||
      !Expect(compiled_half_pack_exponent_program[0].opcode ==
                  Gfx1201CompiledOpcode::kVCvtPkRtzF16F32,
              "expected compiled V_CVT_PK_RTZ_F16_F32 opcode") ||
      !Expect(compiled_half_pack_exponent_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVLdexpF16,
              "expected compiled V_LDEXP_F16 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_half_pack_exponent_state;
  initialize_half_pack_exponent_state(&compiled_half_pack_exponent_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_half_pack_exponent_program,
                                         &compiled_half_pack_exponent_state,
                                         &error_message),
              "expected compiled half pack/exponent execution success") ||
      !Expect(ExpectHalfPackExponentSeedState(
                  compiled_half_pack_exponent_state),
              "expected compiled half pack/exponent state")) {
    return 1;
  }

  const std::array<std::uint32_t, 5> fp8_bridge_words{
      MakeVop1(108u, 103u, 257u),
      MakeVop1(109u, 104u, 258u),
      MakeVop1(110u, 105u, 259u),
      MakeVop1(111u, 107u, 260u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> fp8_bridge_program;
  if (!Expect(decoder.DecodeProgram(fp8_bridge_words, &fp8_bridge_program,
                                    &error_message),
              "expected FP8/BF8 bridge program decode success") ||
      !Expect(fp8_bridge_program.size() == 5u,
              "expected five decoded FP8/BF8 bridge instructions") ||
      !Expect(fp8_bridge_program[0].opcode == "V_CVT_F32_FP8",
              "expected decoded V_CVT_F32_FP8") ||
      !Expect(fp8_bridge_program[1].opcode == "V_CVT_F32_BF8",
              "expected decoded V_CVT_F32_BF8") ||
      !Expect(fp8_bridge_program[2].opcode == "V_CVT_PK_F32_FP8",
              "expected decoded V_CVT_PK_F32_FP8") ||
      !Expect(fp8_bridge_program[3].opcode == "V_CVT_PK_F32_BF8",
              "expected decoded V_CVT_PK_F32_BF8") ||
      !Expect(fp8_bridge_program[4].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after FP8/BF8 bridge batch")) {
    return 1;
  }

  auto initialize_fp8_bridge_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;

    state->vgprs[1][0] = 0x0000003cu;
    state->vgprs[1][1] = 0x000000bcu;
    state->vgprs[1][2] = 0x11111111u;
    state->vgprs[1][3] = 0x0000007fu;

    state->vgprs[2][0] = 0x0000003eu;
    state->vgprs[2][1] = 0x000000beu;
    state->vgprs[2][2] = 0x22222222u;
    state->vgprs[2][3] = 0x0000007cu;

    state->vgprs[3][0] = 0x00007f3cu;
    state->vgprs[3][1] = 0x0000bc3cu;
    state->vgprs[3][2] = 0x33333333u;
    state->vgprs[3][3] = 0x00000000u;

    state->vgprs[4][0] = 0x00007c3eu;
    state->vgprs[4][1] = 0x0000be3eu;
    state->vgprs[4][2] = 0x44444444u;
    state->vgprs[4][3] = 0x0000fc00u;

    state->vgprs[103][2] = 0xa3a3a3a3u;
    state->vgprs[104][2] = 0xa4a4a4a4u;
    state->vgprs[105][2] = 0xa5a5a5a5u;
    state->vgprs[106][2] = 0xa6a6a6a6u;
    state->vgprs[107][2] = 0xa7a7a7a7u;
    state->vgprs[108][2] = 0xa8a8a8a8u;
  };

  auto expect_fp8_bridge_state = [](const WaveExecutionState& state) {
    return state.vgprs[103][0] == FloatBits(1.5f) &&
           state.vgprs[103][1] == FloatBits(-1.5f) &&
           state.vgprs[103][2] == 0xa3a3a3a3u &&
           std::isnan(BitsToFloat(state.vgprs[103][3])) &&
           state.vgprs[104][0] == FloatBits(1.5f) &&
           state.vgprs[104][1] == FloatBits(-1.5f) &&
           state.vgprs[104][2] == 0xa4a4a4a4u &&
           std::isinf(BitsToFloat(state.vgprs[104][3])) &&
           !std::signbit(BitsToFloat(state.vgprs[104][3])) &&
           state.vgprs[105][0] == FloatBits(1.5f) &&
           std::isnan(BitsToFloat(state.vgprs[106][0])) &&
           state.vgprs[105][1] == FloatBits(1.5f) &&
           state.vgprs[106][1] == FloatBits(-1.5f) &&
           state.vgprs[105][2] == 0xa5a5a5a5u &&
           state.vgprs[106][2] == 0xa6a6a6a6u &&
           state.vgprs[105][3] == FloatBits(0.0f) &&
           state.vgprs[106][3] == FloatBits(0.0f) &&
           state.vgprs[107][0] == FloatBits(1.5f) &&
           std::isinf(BitsToFloat(state.vgprs[108][0])) &&
           !std::signbit(BitsToFloat(state.vgprs[108][0])) &&
           state.vgprs[107][1] == FloatBits(1.5f) &&
           state.vgprs[108][1] == FloatBits(-1.5f) &&
           state.vgprs[107][2] == 0xa7a7a7a7u &&
           state.vgprs[108][2] == 0xa8a8a8a8u &&
           state.vgprs[107][3] == FloatBits(0.0f) &&
           std::isinf(BitsToFloat(state.vgprs[108][3])) &&
           std::signbit(BitsToFloat(state.vgprs[108][3]));
  };

  WaveExecutionState decoded_fp8_bridge_state;
  initialize_fp8_bridge_state(&decoded_fp8_bridge_state);
  if (!Expect(interpreter.ExecuteProgram(fp8_bridge_program,
                                         &decoded_fp8_bridge_state,
                                         &error_message),
              "expected decoded FP8/BF8 bridge execution success") ||
      !Expect(expect_fp8_bridge_state(decoded_fp8_bridge_state),
              "expected decoded FP8/BF8 bridge state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_fp8_bridge_program;
  if (!Expect(interpreter.CompileProgram(fp8_bridge_program,
                                         &compiled_fp8_bridge_program,
                                         &error_message),
              "expected compiled FP8/BF8 bridge program success") ||
      !Expect(compiled_fp8_bridge_program.size() == 5u,
              "expected five compiled FP8/BF8 bridge instructions") ||
      !Expect(compiled_fp8_bridge_program[0].opcode ==
                  Gfx1201CompiledOpcode::kVCvtF32Fp8,
              "expected compiled V_CVT_F32_FP8 opcode") ||
      !Expect(compiled_fp8_bridge_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVCvtF32Bf8,
              "expected compiled V_CVT_F32_BF8 opcode") ||
      !Expect(compiled_fp8_bridge_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVCvtPkF32Fp8,
              "expected compiled V_CVT_PK_F32_FP8 opcode") ||
      !Expect(compiled_fp8_bridge_program[3].opcode ==
                  Gfx1201CompiledOpcode::kVCvtPkF32Bf8,
              "expected compiled V_CVT_PK_F32_BF8 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_fp8_bridge_state;
  initialize_fp8_bridge_state(&compiled_fp8_bridge_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_fp8_bridge_program,
                                         &compiled_fp8_bridge_state,
                                         &error_message),
              "expected compiled FP8/BF8 bridge execution success") ||
      !Expect(expect_fp8_bridge_state(compiled_fp8_bridge_state),
              "expected compiled FP8/BF8 bridge state")) {
    return 1;
  }

  if (!RunDx9FmacBatchTest(decoder, interpreter, &error_message)) {
    return 1;
  }
  if (!RunPackedFmacBatchTest(decoder, interpreter, &error_message)) {
    return 1;
  }
  if (!RunHalfLiteralFmaBatchTest(decoder, interpreter, &error_message)) {
    return 1;
  }
  if (!RunCarryChainBatchTest(decoder, interpreter, &error_message)) {
    return 1;
  }

  const std::array<std::uint32_t, 7> f32_vector_binary_words{
      MakeVop2(3u, 103u, 257u, 2u),
      MakeVop2(4u, 104u, 257u, 2u),
      MakeVop2(5u, 105u, 257u, 2u),
      MakeVop2(8u, 106u, 257u, 2u),
      MakeVop2(21u, 107u, 257u, 2u),
      MakeVop2(22u, 108u, 257u, 2u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> f32_vector_binary_program;
  if (!Expect(decoder.DecodeProgram(f32_vector_binary_words,
                                    &f32_vector_binary_program,
                                    &error_message),
              "expected F32 vector binary program decode success") ||
      !Expect(f32_vector_binary_program.size() == 7u,
              "expected seven decoded F32 vector binary instructions") ||
      !Expect(f32_vector_binary_program[0].opcode == "V_ADD_F32",
              "expected decoded V_ADD_F32") ||
      !Expect(f32_vector_binary_program[1].opcode == "V_SUB_F32",
              "expected decoded V_SUB_F32") ||
      !Expect(f32_vector_binary_program[2].opcode == "V_SUBREV_F32",
              "expected decoded V_SUBREV_F32") ||
      !Expect(f32_vector_binary_program[3].opcode == "V_MUL_F32",
              "expected decoded V_MUL_F32") ||
      !Expect(f32_vector_binary_program[4].opcode == "V_MIN_NUM_F32",
              "expected decoded V_MIN_NUM_F32") ||
      !Expect(f32_vector_binary_program[5].opcode == "V_MAX_NUM_F32",
              "expected decoded V_MAX_NUM_F32") ||
      !Expect(f32_vector_binary_program[6].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after F32 vector binary batch")) {
    return 1;
  }

  auto initialize_f32_vector_binary_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;

    state->vgprs[1][0] = FloatBits(1.5f);
    state->vgprs[1][1] = FloatBits(-2.0f);
    state->vgprs[1][2] = 0x11111111u;
    state->vgprs[1][3] = FloatBits(4.0f);

    state->vgprs[2][0] = FloatBits(0.5f);
    state->vgprs[2][1] = FloatBits(3.0f);
    state->vgprs[2][2] = 0x22222222u;
    state->vgprs[2][3] = FloatBits(-1.0f);

    state->vgprs[103][2] = 0xa3a3a3a3u;
    state->vgprs[104][2] = 0xa4a4a4a4u;
    state->vgprs[105][2] = 0xa5a5a5a5u;
    state->vgprs[106][2] = 0xa6a6a6a6u;
    state->vgprs[107][2] = 0xa7a7a7a7u;
    state->vgprs[108][2] = 0xa8a8a8a8u;
  };

  WaveExecutionState decoded_f32_vector_binary_state;
  initialize_f32_vector_binary_state(&decoded_f32_vector_binary_state);
  if (!Expect(interpreter.ExecuteProgram(f32_vector_binary_program,
                                         &decoded_f32_vector_binary_state,
                                         &error_message),
              "expected decoded F32 vector binary execution success") ||
      !Expect(ExpectF32VectorBinarySeedState(
                  decoded_f32_vector_binary_state),
              "expected decoded F32 vector binary state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_f32_vector_binary_program;
  if (!Expect(interpreter.CompileProgram(f32_vector_binary_program,
                                         &compiled_f32_vector_binary_program,
                                         &error_message),
              "expected compiled F32 vector binary program success") ||
      !Expect(compiled_f32_vector_binary_program.size() == 7u,
              "expected seven compiled F32 vector binary instructions") ||
      !Expect(compiled_f32_vector_binary_program[0].opcode ==
                  Gfx1201CompiledOpcode::kVAddF32,
              "expected compiled V_ADD_F32 opcode") ||
      !Expect(compiled_f32_vector_binary_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVSubF32,
              "expected compiled V_SUB_F32 opcode") ||
      !Expect(compiled_f32_vector_binary_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVSubrevF32,
              "expected compiled V_SUBREV_F32 opcode") ||
      !Expect(compiled_f32_vector_binary_program[3].opcode ==
                  Gfx1201CompiledOpcode::kVMulF32,
              "expected compiled V_MUL_F32 opcode") ||
      !Expect(compiled_f32_vector_binary_program[4].opcode ==
                  Gfx1201CompiledOpcode::kVMinNumF32,
              "expected compiled V_MIN_NUM_F32 opcode") ||
      !Expect(compiled_f32_vector_binary_program[5].opcode ==
                  Gfx1201CompiledOpcode::kVMaxNumF32,
              "expected compiled V_MAX_NUM_F32 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_f32_vector_binary_state;
  initialize_f32_vector_binary_state(&compiled_f32_vector_binary_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_f32_vector_binary_program,
                                         &compiled_f32_vector_binary_state,
                                         &error_message),
              "expected compiled F32 vector binary execution success") ||
      !Expect(ExpectF32VectorBinarySeedState(
                  compiled_f32_vector_binary_state),
              "expected compiled F32 vector binary state")) {
    return 1;
  }

  const std::array<std::uint32_t, 5> f64_vector_binary_words{
      MakeVop2(2u, 109u, 257u, 3u),
      MakeVop2(6u, 111u, 257u, 3u),
      MakeVop2(13u, 113u, 257u, 3u),
      MakeVop2(14u, 115u, 257u, 3u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> f64_vector_binary_program;
  if (!Expect(decoder.DecodeProgram(f64_vector_binary_words,
                                    &f64_vector_binary_program,
                                    &error_message),
              "expected F64 vector binary program decode success") ||
      !Expect(f64_vector_binary_program.size() == 5u,
              "expected five decoded F64 vector binary instructions") ||
      !Expect(f64_vector_binary_program[0].opcode == "V_ADD_F64",
              "expected decoded V_ADD_F64") ||
      !Expect(f64_vector_binary_program[1].opcode == "V_MUL_F64",
              "expected decoded V_MUL_F64") ||
      !Expect(f64_vector_binary_program[2].opcode == "V_MIN_NUM_F64",
              "expected decoded V_MIN_NUM_F64") ||
      !Expect(f64_vector_binary_program[3].opcode == "V_MAX_NUM_F64",
              "expected decoded V_MAX_NUM_F64") ||
      !Expect(f64_vector_binary_program[4].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after F64 vector binary batch")) {
    return 1;
  }

  auto initialize_f64_vector_binary_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;

    SplitU64(DoubleBits(1.5), &state->vgprs[1][0], &state->vgprs[2][0]);
    SplitU64(DoubleBits(-2.0), &state->vgprs[1][1], &state->vgprs[2][1]);
    state->vgprs[1][2] = 0x11111111u;
    state->vgprs[2][2] = 0x21212121u;
    SplitU64(DoubleBits(4.0), &state->vgprs[1][3], &state->vgprs[2][3]);

    SplitU64(DoubleBits(0.5), &state->vgprs[3][0], &state->vgprs[4][0]);
    SplitU64(DoubleBits(3.0), &state->vgprs[3][1], &state->vgprs[4][1]);
    state->vgprs[3][2] = 0x33333333u;
    state->vgprs[4][2] = 0x43434343u;
    SplitU64(DoubleBits(-1.0), &state->vgprs[3][3], &state->vgprs[4][3]);

    state->vgprs[109][2] = 0xb9b9b9b9u;
    state->vgprs[110][2] = 0xc9c9c9c9u;
    state->vgprs[111][2] = 0xb1b1b1b1u;
    state->vgprs[112][2] = 0xc1c1c1c1u;
    state->vgprs[113][2] = 0xb3b3b3b3u;
    state->vgprs[114][2] = 0xc3c3c3c3u;
    state->vgprs[115][2] = 0xb5b5b5b5u;
    state->vgprs[116][2] = 0xc5c5c5c5u;
  };

  WaveExecutionState decoded_f64_vector_binary_state;
  initialize_f64_vector_binary_state(&decoded_f64_vector_binary_state);
  if (!Expect(interpreter.ExecuteProgram(f64_vector_binary_program,
                                         &decoded_f64_vector_binary_state,
                                         &error_message),
              "expected decoded F64 vector binary execution success") ||
      !Expect(ExpectF64VectorBinarySeedState(
                  decoded_f64_vector_binary_state),
              "expected decoded F64 vector binary state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_f64_vector_binary_program;
  if (!Expect(interpreter.CompileProgram(f64_vector_binary_program,
                                         &compiled_f64_vector_binary_program,
                                         &error_message),
              "expected compiled F64 vector binary program success") ||
      !Expect(compiled_f64_vector_binary_program.size() == 5u,
              "expected five compiled F64 vector binary instructions") ||
      !Expect(compiled_f64_vector_binary_program[0].opcode ==
                  Gfx1201CompiledOpcode::kVAddF64,
              "expected compiled V_ADD_F64 opcode") ||
      !Expect(compiled_f64_vector_binary_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVMulF64,
              "expected compiled V_MUL_F64 opcode") ||
      !Expect(compiled_f64_vector_binary_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVMinNumF64,
              "expected compiled V_MIN_NUM_F64 opcode") ||
      !Expect(compiled_f64_vector_binary_program[3].opcode ==
                  Gfx1201CompiledOpcode::kVMaxNumF64,
              "expected compiled V_MAX_NUM_F64 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_f64_vector_binary_state;
  initialize_f64_vector_binary_state(&compiled_f64_vector_binary_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_f64_vector_binary_program,
                                         &compiled_f64_vector_binary_state,
                                         &error_message),
              "expected compiled F64 vector binary execution success") ||
      !Expect(ExpectF64VectorBinarySeedState(
                  compiled_f64_vector_binary_state),
              "expected compiled F64 vector binary state")) {
    return 1;
  }

  const std::array<std::uint32_t, 6> i24_vector_binary_words{
      MakeVop2(30u, 117u, 261u, 6u),
      MakeVop2(9u, 118u, 261u, 6u),
      MakeVop2(10u, 119u, 261u, 6u),
      MakeVop2(11u, 120u, 261u, 6u),
      MakeVop2(12u, 121u, 261u, 6u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> i24_vector_binary_program;
  if (!Expect(decoder.DecodeProgram(i24_vector_binary_words,
                                    &i24_vector_binary_program,
                                    &error_message),
              "expected I24 vector binary program decode success") ||
      !Expect(i24_vector_binary_program.size() == 6u,
              "expected six decoded I24 vector binary instructions") ||
      !Expect(i24_vector_binary_program[0].opcode == "V_XNOR_B32",
              "expected decoded V_XNOR_B32") ||
      !Expect(i24_vector_binary_program[1].opcode == "V_MUL_I32_I24",
              "expected decoded V_MUL_I32_I24") ||
      !Expect(i24_vector_binary_program[2].opcode == "V_MUL_HI_I32_I24",
              "expected decoded V_MUL_HI_I32_I24") ||
      !Expect(i24_vector_binary_program[3].opcode == "V_MUL_U32_U24",
              "expected decoded V_MUL_U32_U24") ||
      !Expect(i24_vector_binary_program[4].opcode == "V_MUL_HI_U32_U24",
              "expected decoded V_MUL_HI_U32_U24") ||
      !Expect(i24_vector_binary_program[5].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after I24 vector binary batch")) {
    return 1;
  }

  auto initialize_i24_vector_binary_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;

    state->vgprs[5][0] = 0x00010000u;
    state->vgprs[5][1] = 0x00fffffeu;
    state->vgprs[5][2] = 0x55555555u;
    state->vgprs[5][3] = 0x007fffffu;

    state->vgprs[6][0] = 0x00000010u;
    state->vgprs[6][1] = 0x00000003u;
    state->vgprs[6][2] = 0x66666666u;
    state->vgprs[6][3] = 0x007fffffu;

    state->vgprs[117][2] = 0xd7d7d7d7u;
    state->vgprs[118][2] = 0xd8d8d8d8u;
    state->vgprs[119][2] = 0xd9d9d9d9u;
    state->vgprs[120][2] = 0xdadadadau;
    state->vgprs[121][2] = 0xdbdbdbdbu;
  };

  WaveExecutionState decoded_i24_vector_binary_state;
  initialize_i24_vector_binary_state(&decoded_i24_vector_binary_state);
  if (!Expect(interpreter.ExecuteProgram(i24_vector_binary_program,
                                         &decoded_i24_vector_binary_state,
                                         &error_message),
              "expected decoded I24 vector binary execution success") ||
      !Expect(ExpectI24VectorBinarySeedState(
                  decoded_i24_vector_binary_state),
              "expected decoded I24 vector binary state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_i24_vector_binary_program;
  if (!Expect(interpreter.CompileProgram(i24_vector_binary_program,
                                         &compiled_i24_vector_binary_program,
                                         &error_message),
              "expected compiled I24 vector binary program success") ||
      !Expect(compiled_i24_vector_binary_program.size() == 6u,
              "expected six compiled I24 vector binary instructions") ||
      !Expect(compiled_i24_vector_binary_program[0].opcode ==
                  Gfx1201CompiledOpcode::kVXnorB32,
              "expected compiled V_XNOR_B32 opcode") ||
      !Expect(compiled_i24_vector_binary_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVMulI32I24,
              "expected compiled V_MUL_I32_I24 opcode") ||
      !Expect(compiled_i24_vector_binary_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVMulHiI32I24,
              "expected compiled V_MUL_HI_I32_I24 opcode") ||
      !Expect(compiled_i24_vector_binary_program[3].opcode ==
                  Gfx1201CompiledOpcode::kVMulU32U24,
              "expected compiled V_MUL_U32_U24 opcode") ||
      !Expect(compiled_i24_vector_binary_program[4].opcode ==
                  Gfx1201CompiledOpcode::kVMulHiU32U24,
              "expected compiled V_MUL_HI_U32_U24 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_i24_vector_binary_state;
  initialize_i24_vector_binary_state(&compiled_i24_vector_binary_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_i24_vector_binary_program,
                                         &compiled_i24_vector_binary_state,
                                         &error_message),
              "expected compiled I24 vector binary execution success") ||
      !Expect(ExpectI24VectorBinarySeedState(
                  compiled_i24_vector_binary_state),
              "expected compiled I24 vector binary state")) {
    return 1;
  }

  const auto dcache_inv_words = MakeSmem(33u, 0u, 0u, true, 0u);
  DecodedInstruction dcache_inv_instruction;
  std::size_t dcache_inv_words_consumed = 0;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(dcache_inv_words.data(),
                                                 dcache_inv_words.size()),
                  &dcache_inv_instruction, &dcache_inv_words_consumed,
                  &error_message),
              "expected S_DCACHE_INV direct decode success") ||
      !Expect(dcache_inv_instruction.opcode == "S_DCACHE_INV",
              "expected decoded S_DCACHE_INV opcode") ||
      !Expect(dcache_inv_instruction.operand_count == 0u,
              "expected S_DCACHE_INV nullary decode") ||
      !Expect(dcache_inv_words_consumed == 2u,
              "expected S_DCACHE_INV to consume two dwords")) {
    return 1;
  }

  const std::array<std::uint32_t, 3> dcache_inv_program_words{
      dcache_inv_words[0],
      dcache_inv_words[1],
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> dcache_inv_program;
  if (!Expect(decoder.DecodeProgram(dcache_inv_program_words, &dcache_inv_program,
                                    &error_message),
              "expected S_DCACHE_INV program decode success") ||
      !Expect(dcache_inv_program.size() == 2u,
              "expected two decoded S_DCACHE_INV program instructions") ||
      !Expect(dcache_inv_program[0].opcode == "S_DCACHE_INV",
              "expected decoded S_DCACHE_INV program opcode") ||
      !Expect(dcache_inv_program[1].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after S_DCACHE_INV")) {
    return 1;
  }

  auto initialize_dcache_inv_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;
    state->sgprs[8] = 0x12345678u;
    state->vgprs[3][0] = 0xaabbccddu;
    state->vgprs[3][2] = 0x11223344u;
  };
  auto expect_dcache_inv_state = [](const WaveExecutionState& state) {
    return state.lane_count == 32u && state.exec_mask == 0xbu &&
           state.sgprs[8] == 0x12345678u &&
           state.vgprs[3][0] == 0xaabbccddu &&
           state.vgprs[3][2] == 0x11223344u && state.halted &&
           !state.waiting_on_barrier && state.pc == 1u;
  };

  WaveExecutionState decoded_dcache_inv_state;
  initialize_dcache_inv_state(&decoded_dcache_inv_state);
  if (!Expect(interpreter.ExecuteProgram(dcache_inv_program,
                                         &decoded_dcache_inv_state,
                                         &error_message),
              "expected decoded S_DCACHE_INV execution success") ||
      !Expect(expect_dcache_inv_state(decoded_dcache_inv_state),
              "expected decoded S_DCACHE_INV state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_dcache_inv_program;
  if (!Expect(interpreter.CompileProgram(dcache_inv_program,
                                         &compiled_dcache_inv_program,
                                         &error_message),
              "expected compiled S_DCACHE_INV program success") ||
      !Expect(compiled_dcache_inv_program.size() == 2u,
              "expected two compiled S_DCACHE_INV program instructions") ||
      !Expect(compiled_dcache_inv_program[0].opcode ==
                  Gfx1201CompiledOpcode::kSNop,
              "expected compiled S_DCACHE_INV opcode") ||
      !Expect(compiled_dcache_inv_program[1].opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after S_DCACHE_INV")) {
    return 1;
  }

  WaveExecutionState compiled_dcache_inv_state;
  initialize_dcache_inv_state(&compiled_dcache_inv_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_dcache_inv_program,
                                         &compiled_dcache_inv_state,
                                         &error_message),
              "expected compiled S_DCACHE_INV execution success") ||
      !Expect(expect_dcache_inv_state(compiled_dcache_inv_state),
              "expected compiled S_DCACHE_INV state")) {
    return 1;
  }

  const auto global_inv_words = MakeGlobal(43u, 0u, 0u, 0u, 0u, 0);
  const auto global_wb_words = MakeGlobal(44u, 0u, 0u, 0u, 0u, 0);
  const auto global_wbinv_words = MakeGlobal(79u, 0u, 0u, 0u, 0u, 0);
  const auto global_load_u8_words = MakeGlobal(16u, 20u, 40u, 0u, 20u, 0x000);
  const auto global_load_i8_words = MakeGlobal(17u, 21u, 40u, 0u, 20u, 0x200);
  const auto global_load_u16_words =
      MakeGlobal(18u, 22u, 40u, 0u, 20u, 0x400);
  const auto global_load_i16_words =
      MakeGlobal(19u, 23u, 40u, 0u, 20u, 0x600);
  const auto global_load_b32_words = MakeGlobal(20u, 24u, 40u, 0u, 20u, 0x800);
  const auto global_load_b64_words = MakeGlobal(21u, 25u, 40u, 0u, 20u, 0xa00);
  const auto global_load_b96_words = MakeGlobal(22u, 27u, 40u, 0u, 20u, 0xc00);
  const auto global_load_b128_words =
      MakeGlobal(23u, 30u, 40u, 0u, 20u, 0xe00);
  DecodedInstruction global_inv_instruction;
  std::size_t global_words_consumed = 0;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(global_inv_words.data(),
                                                 global_inv_words.size()),
                  &global_inv_instruction, &global_words_consumed,
                  &error_message),
              "expected GLOBAL_INV direct decode success") ||
      !Expect(global_inv_instruction.opcode == "GLOBAL_INV",
              "expected decoded GLOBAL_INV opcode") ||
      !Expect(global_inv_instruction.operand_count == 0u,
              "expected GLOBAL_INV nullary decode") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_INV to consume two dwords")) {
    return 1;
  }

  DecodedInstruction global_load_b32_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(global_load_b32_words.data(),
                                                 global_load_b32_words.size()),
                  &global_load_b32_instruction, &global_words_consumed,
                  &error_message),
              "expected GLOBAL_LOAD_B32 direct decode success") ||
      !Expect(global_load_b32_instruction.opcode == "GLOBAL_LOAD_B32",
              "expected decoded GLOBAL_LOAD_B32 opcode") ||
      !Expect(global_load_b32_instruction.operand_count == 4u,
              "expected GLOBAL_LOAD_B32 operand count") ||
      !Expect(global_load_b32_instruction.operands[0].kind == OperandKind::kVgpr &&
                  global_load_b32_instruction.operands[0].index == 24u,
              "expected GLOBAL_LOAD_B32 vector destination") ||
      !Expect(global_load_b32_instruction.operands[1].kind == OperandKind::kVgpr &&
                  global_load_b32_instruction.operands[1].index == 40u,
              "expected GLOBAL_LOAD_B32 vector address") ||
      !Expect(global_load_b32_instruction.operands[2].kind == OperandKind::kSgpr &&
                  global_load_b32_instruction.operands[2].index == 20u,
              "expected GLOBAL_LOAD_B32 scalar address") ||
      !Expect(global_load_b32_instruction.operands[3].kind == OperandKind::kImm32 &&
                  global_load_b32_instruction.operands[3].imm32 == 0x800u,
              "expected GLOBAL_LOAD_B32 inline offset") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_LOAD_B32 to consume two dwords")) {
    return 1;
  }

  DecodedInstruction global_load_b128_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(global_load_b128_words.data(),
                                                 global_load_b128_words.size()),
                  &global_load_b128_instruction, &global_words_consumed,
                  &error_message),
              "expected GLOBAL_LOAD_B128 direct decode success") ||
      !Expect(global_load_b128_instruction.opcode == "GLOBAL_LOAD_B128",
              "expected decoded GLOBAL_LOAD_B128 opcode") ||
      !Expect(global_load_b128_instruction.operand_count == 4u,
              "expected GLOBAL_LOAD_B128 operand count") ||
      !Expect(global_load_b128_instruction.operands[0].kind == OperandKind::kVgpr &&
                  global_load_b128_instruction.operands[0].index == 30u,
              "expected GLOBAL_LOAD_B128 vector destination") ||
      !Expect(global_load_b128_instruction.operands[1].kind == OperandKind::kVgpr &&
                  global_load_b128_instruction.operands[1].index == 40u,
              "expected GLOBAL_LOAD_B128 vector address") ||
      !Expect(global_load_b128_instruction.operands[2].kind == OperandKind::kSgpr &&
                  global_load_b128_instruction.operands[2].index == 20u,
              "expected GLOBAL_LOAD_B128 scalar address") ||
      !Expect(global_load_b128_instruction.operands[3].kind == OperandKind::kImm32 &&
                  global_load_b128_instruction.operands[3].imm32 == 0xe00u,
              "expected GLOBAL_LOAD_B128 inline offset") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_LOAD_B128 to consume two dwords")) {
    return 1;
  }

  const auto global_load_d16_u8_words =
      MakeGlobal(30u, 34u, 42u, 0u, 20u, 0x120);
  DecodedInstruction global_load_d16_u8_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_load_d16_u8_words.data(),
                      global_load_d16_u8_words.size()),
                  &global_load_d16_u8_instruction, &global_words_consumed,
                  &error_message),
              "expected GLOBAL_LOAD_D16_U8 direct decode success") ||
      !Expect(global_load_d16_u8_instruction.opcode == "GLOBAL_LOAD_D16_U8",
              "expected decoded GLOBAL_LOAD_D16_U8 opcode") ||
      !Expect(global_load_d16_u8_instruction.operand_count == 4u,
              "expected GLOBAL_LOAD_D16_U8 operand count") ||
      !Expect(global_load_d16_u8_instruction.operands[0].kind ==
                  OperandKind::kVgpr &&
                  global_load_d16_u8_instruction.operands[0].index == 34u,
              "expected GLOBAL_LOAD_D16_U8 vector destination") ||
      !Expect(global_load_d16_u8_instruction.operands[1].kind ==
                  OperandKind::kVgpr &&
                  global_load_d16_u8_instruction.operands[1].index == 42u,
              "expected GLOBAL_LOAD_D16_U8 vector address") ||
      !Expect(global_load_d16_u8_instruction.operands[2].kind ==
                  OperandKind::kSgpr &&
                  global_load_d16_u8_instruction.operands[2].index == 20u,
              "expected GLOBAL_LOAD_D16_U8 scalar address") ||
      !Expect(global_load_d16_u8_instruction.operands[3].kind ==
                  OperandKind::kImm32 &&
                  global_load_d16_u8_instruction.operands[3].imm32 == 0x120u,
              "expected GLOBAL_LOAD_D16_U8 inline offset") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_LOAD_D16_U8 to consume two dwords")) {
    return 1;
  }

  const auto global_load_d16_hi_b16_words =
      MakeGlobal(35u, 35u, 43u, 0u, 21u, 0x340);
  DecodedInstruction global_load_d16_hi_b16_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_load_d16_hi_b16_words.data(),
                      global_load_d16_hi_b16_words.size()),
                  &global_load_d16_hi_b16_instruction, &global_words_consumed,
                  &error_message),
              "expected GLOBAL_LOAD_D16_HI_B16 direct decode success") ||
      !Expect(global_load_d16_hi_b16_instruction.opcode ==
                  "GLOBAL_LOAD_D16_HI_B16",
              "expected decoded GLOBAL_LOAD_D16_HI_B16 opcode") ||
      !Expect(global_load_d16_hi_b16_instruction.operand_count == 4u,
              "expected GLOBAL_LOAD_D16_HI_B16 operand count") ||
      !Expect(global_load_d16_hi_b16_instruction.operands[0].kind ==
                  OperandKind::kVgpr &&
                  global_load_d16_hi_b16_instruction.operands[0].index == 35u,
              "expected GLOBAL_LOAD_D16_HI_B16 vector destination") ||
      !Expect(global_load_d16_hi_b16_instruction.operands[1].kind ==
                  OperandKind::kVgpr &&
                  global_load_d16_hi_b16_instruction.operands[1].index == 43u,
              "expected GLOBAL_LOAD_D16_HI_B16 vector address") ||
      !Expect(global_load_d16_hi_b16_instruction.operands[2].kind ==
                  OperandKind::kSgpr &&
                  global_load_d16_hi_b16_instruction.operands[2].index == 21u,
              "expected GLOBAL_LOAD_D16_HI_B16 scalar address") ||
      !Expect(global_load_d16_hi_b16_instruction.operands[3].kind ==
                  OperandKind::kImm32 &&
                  global_load_d16_hi_b16_instruction.operands[3].imm32 ==
                      0x340u,
              "expected GLOBAL_LOAD_D16_HI_B16 inline offset") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_LOAD_D16_HI_B16 to consume two dwords")) {
    return 1;
  }

  const auto global_load_tr_b64_words =
      MakeGlobal(88u, 36u, 44u, 0u, 22u, 0x460);
  DecodedInstruction global_load_tr_b64_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_load_tr_b64_words.data(),
                      global_load_tr_b64_words.size()),
                  &global_load_tr_b64_instruction, &global_words_consumed,
                  &error_message),
              "expected GLOBAL_LOAD_TR_B64 direct decode success") ||
      !Expect(global_load_tr_b64_instruction.opcode == "GLOBAL_LOAD_TR_B64",
              "expected decoded GLOBAL_LOAD_TR_B64 opcode") ||
      !Expect(global_load_tr_b64_instruction.operand_count == 4u,
              "expected GLOBAL_LOAD_TR_B64 operand count") ||
      !Expect(global_load_tr_b64_instruction.operands[0].kind ==
                  OperandKind::kVgpr &&
                  global_load_tr_b64_instruction.operands[0].index == 36u,
              "expected GLOBAL_LOAD_TR_B64 vector destination") ||
      !Expect(global_load_tr_b64_instruction.operands[1].kind ==
                  OperandKind::kVgpr &&
                  global_load_tr_b64_instruction.operands[1].index == 44u,
              "expected GLOBAL_LOAD_TR_B64 vector address") ||
      !Expect(global_load_tr_b64_instruction.operands[2].kind ==
                  OperandKind::kSgpr &&
                  global_load_tr_b64_instruction.operands[2].index == 22u,
              "expected GLOBAL_LOAD_TR_B64 scalar address") ||
      !Expect(global_load_tr_b64_instruction.operands[3].kind ==
                  OperandKind::kImm32 &&
                  global_load_tr_b64_instruction.operands[3].imm32 == 0x460u,
              "expected GLOBAL_LOAD_TR_B64 inline offset") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_LOAD_TR_B64 to consume two dwords")) {
    return 1;
  }

  const auto global_load_tr_b128_words =
      MakeGlobal(87u, 38u, 45u, 0u, 23u, 0x680);
  DecodedInstruction global_load_tr_b128_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_load_tr_b128_words.data(),
                      global_load_tr_b128_words.size()),
                  &global_load_tr_b128_instruction, &global_words_consumed,
                  &error_message),
              "expected GLOBAL_LOAD_TR_B128 direct decode success") ||
      !Expect(global_load_tr_b128_instruction.opcode == "GLOBAL_LOAD_TR_B128",
              "expected decoded GLOBAL_LOAD_TR_B128 opcode") ||
      !Expect(global_load_tr_b128_instruction.operand_count == 4u,
              "expected GLOBAL_LOAD_TR_B128 operand count") ||
      !Expect(global_load_tr_b128_instruction.operands[0].kind ==
                  OperandKind::kVgpr &&
                  global_load_tr_b128_instruction.operands[0].index == 38u,
              "expected GLOBAL_LOAD_TR_B128 vector destination") ||
      !Expect(global_load_tr_b128_instruction.operands[1].kind ==
                  OperandKind::kVgpr &&
                  global_load_tr_b128_instruction.operands[1].index == 45u,
              "expected GLOBAL_LOAD_TR_B128 vector address") ||
      !Expect(global_load_tr_b128_instruction.operands[2].kind ==
                  OperandKind::kSgpr &&
                  global_load_tr_b128_instruction.operands[2].index == 23u,
              "expected GLOBAL_LOAD_TR_B128 scalar address") ||
      !Expect(global_load_tr_b128_instruction.operands[3].kind ==
                  OperandKind::kImm32 &&
                  global_load_tr_b128_instruction.operands[3].imm32 == 0x680u,
              "expected GLOBAL_LOAD_TR_B128 inline offset") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_LOAD_TR_B128 to consume two dwords")) {
    return 1;
  }

  const auto global_load_addtid_b32_words =
      MakeGlobal(40u, 39u, 0u, 0u, 24u, 0);
  DecodedInstruction global_load_addtid_b32_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_load_addtid_b32_words.data(),
                      global_load_addtid_b32_words.size()),
                  &global_load_addtid_b32_instruction, &global_words_consumed,
                  &error_message),
              "expected GLOBAL_LOAD_ADDTID_B32 direct decode success") ||
      !Expect(global_load_addtid_b32_instruction.opcode ==
                  "GLOBAL_LOAD_ADDTID_B32",
              "expected decoded GLOBAL_LOAD_ADDTID_B32 opcode") ||
      !Expect(global_load_addtid_b32_instruction.operand_count == 3u,
              "expected GLOBAL_LOAD_ADDTID_B32 operand count") ||
      !Expect(global_load_addtid_b32_instruction.operands[0].kind ==
                  OperandKind::kVgpr &&
                  global_load_addtid_b32_instruction.operands[0].index == 39u,
              "expected GLOBAL_LOAD_ADDTID_B32 vector destination") ||
      !Expect(global_load_addtid_b32_instruction.operands[1].kind ==
                  OperandKind::kSgpr &&
                  global_load_addtid_b32_instruction.operands[1].index == 24u,
              "expected GLOBAL_LOAD_ADDTID_B32 scalar address") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_LOAD_ADDTID_B32 to consume two dwords")) {
    return 1;
  }

  const auto global_load_block_words =
      MakeGlobal(83u, 40u, 46u, 0u, 25u, 0x120);
  DecodedInstruction global_load_block_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(global_load_block_words.data(),
                                                 global_load_block_words.size()),
                  &global_load_block_instruction, &global_words_consumed,
                  &error_message),
              "expected GLOBAL_LOAD_BLOCK direct decode success") ||
      !Expect(global_load_block_instruction.opcode == "GLOBAL_LOAD_BLOCK",
              "expected decoded GLOBAL_LOAD_BLOCK opcode") ||
      !Expect(global_load_block_instruction.operand_count == 5u,
              "expected GLOBAL_LOAD_BLOCK operand count") ||
      !Expect(global_load_block_instruction.operands[0].kind ==
                  OperandKind::kVgpr &&
                  global_load_block_instruction.operands[0].index == 40u,
              "expected GLOBAL_LOAD_BLOCK vector destination base") ||
      !Expect(global_load_block_instruction.operands[1].kind ==
                  OperandKind::kVgpr &&
                  global_load_block_instruction.operands[1].index == 46u,
              "expected GLOBAL_LOAD_BLOCK vector address") ||
      !Expect(global_load_block_instruction.operands[2].kind ==
                  OperandKind::kSgpr &&
                  global_load_block_instruction.operands[2].index == 25u,
              "expected GLOBAL_LOAD_BLOCK scalar address") ||
      !Expect(global_load_block_instruction.operands[3].kind ==
                  OperandKind::kImm32 &&
                  global_load_block_instruction.operands[3].imm32 == 0x120u,
              "expected GLOBAL_LOAD_BLOCK inline offset") ||
      !Expect(global_load_block_instruction.operands[4].kind ==
                  OperandKind::kSgpr &&
                  global_load_block_instruction.operands[4].index ==
                      kM0RegisterIndex,
              "expected GLOBAL_LOAD_BLOCK implicit M0 source") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_LOAD_BLOCK to consume two dwords")) {
    return 1;
  }

  const auto global_store_b32_words = MakeGlobal(26u, 0u, 44u, 18u, 30u, 16);
  DecodedInstruction global_store_b32_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(global_store_b32_words.data(),
                                                 global_store_b32_words.size()),
                  &global_store_b32_instruction, &global_words_consumed,
                  &error_message),
              "expected GLOBAL_STORE_B32 direct decode success") ||
      !Expect(global_store_b32_instruction.opcode == "GLOBAL_STORE_B32",
              "expected decoded GLOBAL_STORE_B32 opcode") ||
      !Expect(global_store_b32_instruction.operand_count == 4u,
              "expected GLOBAL_STORE_B32 operand count") ||
      !Expect(global_store_b32_instruction.operands[0].kind == OperandKind::kVgpr &&
                  global_store_b32_instruction.operands[0].index == 18u,
              "expected GLOBAL_STORE_B32 vector data") ||
      !Expect(global_store_b32_instruction.operands[1].kind == OperandKind::kVgpr &&
                  global_store_b32_instruction.operands[1].index == 44u,
              "expected GLOBAL_STORE_B32 vector address") ||
      !Expect(global_store_b32_instruction.operands[2].kind == OperandKind::kSgpr &&
                  global_store_b32_instruction.operands[2].index == 30u,
              "expected GLOBAL_STORE_B32 scalar address") ||
      !Expect(global_store_b32_instruction.operands[3].kind == OperandKind::kImm32 &&
                  global_store_b32_instruction.operands[3].imm32 == 16u,
              "expected GLOBAL_STORE_B32 inline offset") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_STORE_B32 to consume two dwords")) {
    return 1;
  }

  const auto global_store_b128_words = MakeGlobal(29u, 0u, 48u, 22u, 32u, 28);
  DecodedInstruction global_store_b128_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(global_store_b128_words.data(),
                                                 global_store_b128_words.size()),
                  &global_store_b128_instruction, &global_words_consumed,
                  &error_message),
              "expected GLOBAL_STORE_B128 direct decode success") ||
      !Expect(global_store_b128_instruction.opcode == "GLOBAL_STORE_B128",
              "expected decoded GLOBAL_STORE_B128 opcode") ||
      !Expect(global_store_b128_instruction.operand_count == 4u,
              "expected GLOBAL_STORE_B128 operand count") ||
      !Expect(global_store_b128_instruction.operands[0].kind == OperandKind::kVgpr &&
                  global_store_b128_instruction.operands[0].index == 22u,
              "expected GLOBAL_STORE_B128 vector data") ||
      !Expect(global_store_b128_instruction.operands[1].kind == OperandKind::kVgpr &&
                  global_store_b128_instruction.operands[1].index == 48u,
              "expected GLOBAL_STORE_B128 vector address") ||
      !Expect(global_store_b128_instruction.operands[2].kind == OperandKind::kSgpr &&
                  global_store_b128_instruction.operands[2].index == 32u,
              "expected GLOBAL_STORE_B128 scalar address") ||
      !Expect(global_store_b128_instruction.operands[3].kind == OperandKind::kImm32 &&
                  global_store_b128_instruction.operands[3].imm32 == 28u,
              "expected GLOBAL_STORE_B128 inline offset") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_STORE_B128 to consume two dwords")) {
    return 1;
  }

  const auto global_store_d16_hi_b8_words =
      MakeGlobal(36u, 0u, 49u, 24u, 33u, 0x44);
  DecodedInstruction global_store_d16_hi_b8_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_store_d16_hi_b8_words.data(),
                      global_store_d16_hi_b8_words.size()),
                  &global_store_d16_hi_b8_instruction, &global_words_consumed,
                  &error_message),
              "expected GLOBAL_STORE_D16_HI_B8 direct decode success") ||
      !Expect(global_store_d16_hi_b8_instruction.opcode ==
                  "GLOBAL_STORE_D16_HI_B8",
              "expected decoded GLOBAL_STORE_D16_HI_B8 opcode") ||
      !Expect(global_store_d16_hi_b8_instruction.operand_count == 4u,
              "expected GLOBAL_STORE_D16_HI_B8 operand count") ||
      !Expect(global_store_d16_hi_b8_instruction.operands[0].kind ==
                  OperandKind::kVgpr &&
                  global_store_d16_hi_b8_instruction.operands[0].index == 24u,
              "expected GLOBAL_STORE_D16_HI_B8 vector data") ||
      !Expect(global_store_d16_hi_b8_instruction.operands[1].kind ==
                  OperandKind::kVgpr &&
                  global_store_d16_hi_b8_instruction.operands[1].index == 49u,
              "expected GLOBAL_STORE_D16_HI_B8 vector address") ||
      !Expect(global_store_d16_hi_b8_instruction.operands[2].kind ==
                  OperandKind::kSgpr &&
                  global_store_d16_hi_b8_instruction.operands[2].index == 33u,
              "expected GLOBAL_STORE_D16_HI_B8 scalar address") ||
      !Expect(global_store_d16_hi_b8_instruction.operands[3].kind ==
                  OperandKind::kImm32 &&
                  global_store_d16_hi_b8_instruction.operands[3].imm32 ==
                      0x44u,
              "expected GLOBAL_STORE_D16_HI_B8 inline offset") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_STORE_D16_HI_B8 to consume two dwords")) {
    return 1;
  }

  const auto global_store_addtid_b32_words =
      MakeGlobal(41u, 0u, 0u, 26u, 34u, 0);
  DecodedInstruction global_store_addtid_b32_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_store_addtid_b32_words.data(),
                      global_store_addtid_b32_words.size()),
                  &global_store_addtid_b32_instruction, &global_words_consumed,
                  &error_message),
              "expected GLOBAL_STORE_ADDTID_B32 direct decode success") ||
      !Expect(global_store_addtid_b32_instruction.opcode ==
                  "GLOBAL_STORE_ADDTID_B32",
              "expected decoded GLOBAL_STORE_ADDTID_B32 opcode") ||
      !Expect(global_store_addtid_b32_instruction.operand_count == 3u,
              "expected GLOBAL_STORE_ADDTID_B32 operand count") ||
      !Expect(global_store_addtid_b32_instruction.operands[0].kind ==
                  OperandKind::kVgpr &&
                  global_store_addtid_b32_instruction.operands[0].index == 26u,
              "expected GLOBAL_STORE_ADDTID_B32 vector data") ||
      !Expect(global_store_addtid_b32_instruction.operands[1].kind ==
                  OperandKind::kSgpr &&
                  global_store_addtid_b32_instruction.operands[1].index == 34u,
              "expected GLOBAL_STORE_ADDTID_B32 scalar address") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_STORE_ADDTID_B32 to consume two dwords")) {
    return 1;
  }

  const auto global_store_block_words =
      MakeGlobal(84u, 0u, 47u, 28u, 35u, 0x44);
  DecodedInstruction global_store_block_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(global_store_block_words.data(),
                                                 global_store_block_words.size()),
                  &global_store_block_instruction, &global_words_consumed,
                  &error_message),
              "expected GLOBAL_STORE_BLOCK direct decode success") ||
      !Expect(global_store_block_instruction.opcode == "GLOBAL_STORE_BLOCK",
              "expected decoded GLOBAL_STORE_BLOCK opcode") ||
      !Expect(global_store_block_instruction.operand_count == 5u,
              "expected GLOBAL_STORE_BLOCK operand count") ||
      !Expect(global_store_block_instruction.operands[0].kind ==
                  OperandKind::kVgpr &&
                  global_store_block_instruction.operands[0].index == 28u,
              "expected GLOBAL_STORE_BLOCK vector data base") ||
      !Expect(global_store_block_instruction.operands[1].kind ==
                  OperandKind::kVgpr &&
                  global_store_block_instruction.operands[1].index == 47u,
              "expected GLOBAL_STORE_BLOCK vector address") ||
      !Expect(global_store_block_instruction.operands[2].kind ==
                  OperandKind::kSgpr &&
                  global_store_block_instruction.operands[2].index == 35u,
              "expected GLOBAL_STORE_BLOCK scalar address") ||
      !Expect(global_store_block_instruction.operands[3].kind ==
                  OperandKind::kImm32 &&
                  global_store_block_instruction.operands[3].imm32 == 0x44u,
              "expected GLOBAL_STORE_BLOCK inline offset") ||
      !Expect(global_store_block_instruction.operands[4].kind ==
                  OperandKind::kSgpr &&
                  global_store_block_instruction.operands[4].index ==
                      kM0RegisterIndex,
              "expected GLOBAL_STORE_BLOCK implicit M0 source") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_STORE_BLOCK to consume two dwords")) {
    return 1;
  }

  const auto global_atomic_swap_b32_words =
      MakeGlobal(51u, 58u, 59u, 30u, 36u, 0x48);
  DecodedInstruction global_atomic_swap_b32_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_atomic_swap_b32_words.data(),
                      global_atomic_swap_b32_words.size()),
                  &global_atomic_swap_b32_instruction, &global_words_consumed,
                  &error_message),
              "expected GLOBAL_ATOMIC_SWAP_B32 direct decode success") ||
      !Expect(global_atomic_swap_b32_instruction.opcode ==
                  "GLOBAL_ATOMIC_SWAP_B32",
              "expected decoded GLOBAL_ATOMIC_SWAP_B32 opcode") ||
      !Expect(global_atomic_swap_b32_instruction.operand_count == 5u,
              "expected GLOBAL_ATOMIC_SWAP_B32 operand count") ||
      !Expect(global_atomic_swap_b32_instruction.operands[0].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_swap_b32_instruction.operands[0].index == 58u,
              "expected GLOBAL_ATOMIC_SWAP_B32 vector destination") ||
      !Expect(global_atomic_swap_b32_instruction.operands[1].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_swap_b32_instruction.operands[1].index == 30u,
              "expected GLOBAL_ATOMIC_SWAP_B32 vector data") ||
      !Expect(global_atomic_swap_b32_instruction.operands[2].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_swap_b32_instruction.operands[2].index == 59u,
              "expected GLOBAL_ATOMIC_SWAP_B32 vector address") ||
      !Expect(global_atomic_swap_b32_instruction.operands[3].kind ==
                  OperandKind::kSgpr &&
                  global_atomic_swap_b32_instruction.operands[3].index == 36u,
              "expected GLOBAL_ATOMIC_SWAP_B32 scalar address") ||
      !Expect(global_atomic_swap_b32_instruction.operands[4].kind ==
                  OperandKind::kImm32 &&
                  global_atomic_swap_b32_instruction.operands[4].imm32 ==
                      0x48u,
              "expected GLOBAL_ATOMIC_SWAP_B32 inline offset") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_ATOMIC_SWAP_B32 to consume two dwords")) {
    return 1;
  }

  const auto global_atomic_cmpswap_b32_words =
      MakeGlobal(52u, 60u, 61u, 31u, 37u, 0x4c);
  DecodedInstruction global_atomic_cmpswap_b32_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_atomic_cmpswap_b32_words.data(),
                      global_atomic_cmpswap_b32_words.size()),
                  &global_atomic_cmpswap_b32_instruction,
                  &global_words_consumed, &error_message),
              "expected GLOBAL_ATOMIC_CMPSWAP_B32 direct decode success") ||
      !Expect(global_atomic_cmpswap_b32_instruction.opcode ==
                  "GLOBAL_ATOMIC_CMPSWAP_B32",
              "expected decoded GLOBAL_ATOMIC_CMPSWAP_B32 opcode") ||
      !Expect(global_atomic_cmpswap_b32_instruction.operand_count == 5u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B32 operand count") ||
      !Expect(global_atomic_cmpswap_b32_instruction.operands[0].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_cmpswap_b32_instruction.operands[0].index ==
                      60u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B32 vector destination") ||
      !Expect(global_atomic_cmpswap_b32_instruction.operands[1].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_cmpswap_b32_instruction.operands[1].index ==
                      31u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B32 vector data pair base") ||
      !Expect(global_atomic_cmpswap_b32_instruction.operands[2].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_cmpswap_b32_instruction.operands[2].index ==
                      61u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B32 vector address") ||
      !Expect(global_atomic_cmpswap_b32_instruction.operands[3].kind ==
                  OperandKind::kSgpr &&
                  global_atomic_cmpswap_b32_instruction.operands[3].index ==
                      37u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B32 scalar address") ||
      !Expect(global_atomic_cmpswap_b32_instruction.operands[4].kind ==
                  OperandKind::kImm32 &&
                  global_atomic_cmpswap_b32_instruction.operands[4].imm32 ==
                      0x4cu,
              "expected GLOBAL_ATOMIC_CMPSWAP_B32 inline offset") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B32 to consume two dwords")) {
    return 1;
  }

  const auto global_atomic_swap_b64_words =
      MakeGlobal(65u, 62u, 63u, 40u, 38u, 0x50);
  DecodedInstruction global_atomic_swap_b64_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_atomic_swap_b64_words.data(),
                      global_atomic_swap_b64_words.size()),
                  &global_atomic_swap_b64_instruction, &global_words_consumed,
                  &error_message),
              "expected GLOBAL_ATOMIC_SWAP_B64 direct decode success") ||
      !Expect(global_atomic_swap_b64_instruction.opcode ==
                  "GLOBAL_ATOMIC_SWAP_B64",
              "expected decoded GLOBAL_ATOMIC_SWAP_B64 opcode") ||
      !Expect(global_atomic_swap_b64_instruction.operand_count == 5u,
              "expected GLOBAL_ATOMIC_SWAP_B64 operand count") ||
      !Expect(global_atomic_swap_b64_instruction.operands[0].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_swap_b64_instruction.operands[0].index == 62u,
              "expected GLOBAL_ATOMIC_SWAP_B64 vector destination pair") ||
      !Expect(global_atomic_swap_b64_instruction.operands[1].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_swap_b64_instruction.operands[1].index == 40u,
              "expected GLOBAL_ATOMIC_SWAP_B64 vector data pair") ||
      !Expect(global_atomic_swap_b64_instruction.operands[2].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_swap_b64_instruction.operands[2].index == 63u,
              "expected GLOBAL_ATOMIC_SWAP_B64 vector address") ||
      !Expect(global_atomic_swap_b64_instruction.operands[3].kind ==
                  OperandKind::kSgpr &&
                  global_atomic_swap_b64_instruction.operands[3].index == 38u,
              "expected GLOBAL_ATOMIC_SWAP_B64 scalar address") ||
      !Expect(global_atomic_swap_b64_instruction.operands[4].kind ==
                  OperandKind::kImm32 &&
                  global_atomic_swap_b64_instruction.operands[4].imm32 ==
                      0x50u,
              "expected GLOBAL_ATOMIC_SWAP_B64 inline offset") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_ATOMIC_SWAP_B64 to consume two dwords")) {
    return 1;
  }

  const auto global_atomic_cmpswap_b64_words =
      MakeGlobal(66u, 64u, 65u, 42u, 39u, 0x54);
  DecodedInstruction global_atomic_cmpswap_b64_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_atomic_cmpswap_b64_words.data(),
                      global_atomic_cmpswap_b64_words.size()),
                  &global_atomic_cmpswap_b64_instruction,
                  &global_words_consumed, &error_message),
              "expected GLOBAL_ATOMIC_CMPSWAP_B64 direct decode success") ||
      !Expect(global_atomic_cmpswap_b64_instruction.opcode ==
                  "GLOBAL_ATOMIC_CMPSWAP_B64",
              "expected decoded GLOBAL_ATOMIC_CMPSWAP_B64 opcode") ||
      !Expect(global_atomic_cmpswap_b64_instruction.operand_count == 5u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B64 operand count") ||
      !Expect(global_atomic_cmpswap_b64_instruction.operands[0].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_cmpswap_b64_instruction.operands[0].index ==
                      64u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B64 vector destination pair") ||
      !Expect(global_atomic_cmpswap_b64_instruction.operands[1].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_cmpswap_b64_instruction.operands[1].index ==
                      42u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B64 vector data quad base") ||
      !Expect(global_atomic_cmpswap_b64_instruction.operands[2].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_cmpswap_b64_instruction.operands[2].index ==
                      65u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B64 vector address") ||
      !Expect(global_atomic_cmpswap_b64_instruction.operands[3].kind ==
                  OperandKind::kSgpr &&
                  global_atomic_cmpswap_b64_instruction.operands[3].index ==
                      39u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B64 scalar address") ||
      !Expect(global_atomic_cmpswap_b64_instruction.operands[4].kind ==
                  OperandKind::kImm32 &&
                  global_atomic_cmpswap_b64_instruction.operands[4].imm32 ==
                      0x54u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B64 inline offset") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_ATOMIC_CMPSWAP_B64 to consume two dwords")) {
    return 1;
  }

  const auto global_atomic_add_f32_words =
      MakeGlobal(86u, 66u, 67u, 43u, 40u, 0x58);
  DecodedInstruction global_atomic_add_f32_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_atomic_add_f32_words.data(),
                      global_atomic_add_f32_words.size()),
                  &global_atomic_add_f32_instruction, &global_words_consumed,
                  &error_message),
              "expected GLOBAL_ATOMIC_ADD_F32 direct decode success") ||
      !Expect(global_atomic_add_f32_instruction.opcode ==
                  "GLOBAL_ATOMIC_ADD_F32",
              "expected decoded GLOBAL_ATOMIC_ADD_F32 opcode") ||
      !Expect(global_atomic_add_f32_instruction.operand_count == 5u,
              "expected GLOBAL_ATOMIC_ADD_F32 operand count") ||
      !Expect(global_atomic_add_f32_instruction.operands[0].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_add_f32_instruction.operands[0].index == 66u,
              "expected GLOBAL_ATOMIC_ADD_F32 vector destination") ||
      !Expect(global_atomic_add_f32_instruction.operands[1].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_add_f32_instruction.operands[1].index == 43u,
              "expected GLOBAL_ATOMIC_ADD_F32 vector data") ||
      !Expect(global_atomic_add_f32_instruction.operands[2].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_add_f32_instruction.operands[2].index == 67u,
              "expected GLOBAL_ATOMIC_ADD_F32 vector address") ||
      !Expect(global_atomic_add_f32_instruction.operands[3].kind ==
                  OperandKind::kSgpr &&
                  global_atomic_add_f32_instruction.operands[3].index == 40u,
              "expected GLOBAL_ATOMIC_ADD_F32 scalar address") ||
      !Expect(global_atomic_add_f32_instruction.operands[4].kind ==
                  OperandKind::kImm32 &&
                  global_atomic_add_f32_instruction.operands[4].imm32 ==
                      0x58u,
              "expected GLOBAL_ATOMIC_ADD_F32 inline offset") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_ATOMIC_ADD_F32 to consume two dwords")) {
    return 1;
  }

  const auto global_atomic_pk_add_f16_words =
      MakeGlobal(89u, 68u, 69u, 44u, 41u, 0x24);
  DecodedInstruction global_atomic_pk_add_f16_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_atomic_pk_add_f16_words.data(),
                      global_atomic_pk_add_f16_words.size()),
                  &global_atomic_pk_add_f16_instruction, &global_words_consumed,
                  &error_message),
              "expected GLOBAL_ATOMIC_PK_ADD_F16 direct decode success") ||
      !Expect(global_atomic_pk_add_f16_instruction.opcode ==
                  "GLOBAL_ATOMIC_PK_ADD_F16",
              "expected decoded GLOBAL_ATOMIC_PK_ADD_F16 opcode") ||
      !Expect(global_atomic_pk_add_f16_instruction.operand_count == 5u,
              "expected GLOBAL_ATOMIC_PK_ADD_F16 operand count") ||
      !Expect(global_atomic_pk_add_f16_instruction.operands[0].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_pk_add_f16_instruction.operands[0].index == 68u,
              "expected GLOBAL_ATOMIC_PK_ADD_F16 vector destination") ||
      !Expect(global_atomic_pk_add_f16_instruction.operands[1].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_pk_add_f16_instruction.operands[1].index == 44u,
              "expected GLOBAL_ATOMIC_PK_ADD_F16 vector data") ||
      !Expect(global_atomic_pk_add_f16_instruction.operands[2].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_pk_add_f16_instruction.operands[2].index == 69u,
              "expected GLOBAL_ATOMIC_PK_ADD_F16 vector address") ||
      !Expect(global_atomic_pk_add_f16_instruction.operands[3].kind ==
                  OperandKind::kSgpr &&
                  global_atomic_pk_add_f16_instruction.operands[3].index == 41u,
              "expected GLOBAL_ATOMIC_PK_ADD_F16 scalar address") ||
      !Expect(global_atomic_pk_add_f16_instruction.operands[4].kind ==
                  OperandKind::kImm32 &&
                  global_atomic_pk_add_f16_instruction.operands[4].imm32 ==
                      0x24u,
              "expected GLOBAL_ATOMIC_PK_ADD_F16 inline offset") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_ATOMIC_PK_ADD_F16 to consume two dwords")) {
    return 1;
  }

  const auto global_atomic_ordered_add_b64_words =
      MakeGlobal(115u, 70u, 71u, 46u, 42u, 0x28);
  DecodedInstruction global_atomic_ordered_add_b64_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      global_atomic_ordered_add_b64_words.data(),
                      global_atomic_ordered_add_b64_words.size()),
                  &global_atomic_ordered_add_b64_instruction,
                  &global_words_consumed, &error_message),
              "expected GLOBAL_ATOMIC_ORDERED_ADD_B64 direct decode success") ||
      !Expect(global_atomic_ordered_add_b64_instruction.opcode ==
                  "GLOBAL_ATOMIC_ORDERED_ADD_B64",
              "expected decoded GLOBAL_ATOMIC_ORDERED_ADD_B64 opcode") ||
      !Expect(global_atomic_ordered_add_b64_instruction.operand_count == 5u,
              "expected GLOBAL_ATOMIC_ORDERED_ADD_B64 operand count") ||
      !Expect(global_atomic_ordered_add_b64_instruction.operands[0].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_ordered_add_b64_instruction.operands[0].index ==
                      70u,
              "expected GLOBAL_ATOMIC_ORDERED_ADD_B64 vector destination") ||
      !Expect(global_atomic_ordered_add_b64_instruction.operands[1].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_ordered_add_b64_instruction.operands[1].index ==
                      46u,
              "expected GLOBAL_ATOMIC_ORDERED_ADD_B64 vector data") ||
      !Expect(global_atomic_ordered_add_b64_instruction.operands[2].kind ==
                  OperandKind::kVgpr &&
                  global_atomic_ordered_add_b64_instruction.operands[2].index ==
                      71u,
              "expected GLOBAL_ATOMIC_ORDERED_ADD_B64 vector address") ||
      !Expect(global_atomic_ordered_add_b64_instruction.operands[3].kind ==
                  OperandKind::kSgpr &&
                  global_atomic_ordered_add_b64_instruction.operands[3].index ==
                      42u,
              "expected GLOBAL_ATOMIC_ORDERED_ADD_B64 scalar address") ||
      !Expect(global_atomic_ordered_add_b64_instruction.operands[4].kind ==
                  OperandKind::kImm32 &&
                  global_atomic_ordered_add_b64_instruction.operands[4].imm32 ==
                      0x28u,
              "expected GLOBAL_ATOMIC_ORDERED_ADD_B64 inline offset") ||
      !Expect(global_words_consumed == 2u,
              "expected GLOBAL_ATOMIC_ORDERED_ADD_B64 to consume two dwords")) {
    return 1;
  }

  const std::array<std::uint32_t, 7> global_hint_program_words{
      global_inv_words[0], global_inv_words[1], global_wb_words[0],
      global_wb_words[1], global_wbinv_words[0], global_wbinv_words[1],
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> global_hint_program;
  if (!Expect(decoder.DecodeProgram(global_hint_program_words,
                                    &global_hint_program, &error_message),
              "expected GLOBAL hint program decode success") ||
      !Expect(global_hint_program.size() == 4u,
              "expected four decoded GLOBAL hint program instructions") ||
      !Expect(global_hint_program[0].opcode == "GLOBAL_INV",
              "expected decoded GLOBAL_INV program opcode") ||
      !Expect(global_hint_program[1].opcode == "GLOBAL_WB",
              "expected decoded GLOBAL_WB program opcode") ||
      !Expect(global_hint_program[2].opcode == "GLOBAL_WBINV",
              "expected decoded GLOBAL_WBINV program opcode") ||
      !Expect(global_hint_program[3].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after GLOBAL hints")) {
    return 1;
  }

  auto initialize_global_hint_state = [](WaveExecutionState* state) {
    state->exec_mask = 0x9u;
    state->sgprs[8] = 0x12345678u;
    state->vgprs[3][0] = 0xaabbccddu;
    state->vgprs[3][3] = 0x11223344u;
  };
  auto expect_global_hint_state = [](const WaveExecutionState& state) {
    return state.lane_count == 32u && state.exec_mask == 0x9u &&
           state.sgprs[8] == 0x12345678u &&
           state.vgprs[3][0] == 0xaabbccddu &&
           state.vgprs[3][3] == 0x11223344u && state.halted &&
           !state.waiting_on_barrier && state.pc == 3u;
  };

  WaveExecutionState decoded_global_hint_state;
  initialize_global_hint_state(&decoded_global_hint_state);
  if (!Expect(interpreter.ExecuteProgram(global_hint_program,
                                         &decoded_global_hint_state,
                                         &error_message),
              "expected decoded GLOBAL hint execution success") ||
      !Expect(expect_global_hint_state(decoded_global_hint_state),
              "expected decoded GLOBAL hint state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_global_hint_program;
  if (!Expect(interpreter.CompileProgram(global_hint_program,
                                         &compiled_global_hint_program,
                                         &error_message),
              "expected compiled GLOBAL hint program success") ||
      !Expect(compiled_global_hint_program.size() == 4u,
              "expected four compiled GLOBAL hint program instructions") ||
      !Expect(compiled_global_hint_program[0].opcode ==
                  Gfx1201CompiledOpcode::kSNop,
              "expected compiled GLOBAL_INV opcode") ||
      !Expect(compiled_global_hint_program[1].opcode ==
                  Gfx1201CompiledOpcode::kSNop,
              "expected compiled GLOBAL_WB opcode") ||
      !Expect(compiled_global_hint_program[2].opcode ==
                  Gfx1201CompiledOpcode::kSNop,
              "expected compiled GLOBAL_WBINV opcode") ||
      !Expect(compiled_global_hint_program[3].opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after GLOBAL hints")) {
    return 1;
  }

  WaveExecutionState compiled_global_hint_state;
  initialize_global_hint_state(&compiled_global_hint_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_global_hint_program,
                                         &compiled_global_hint_state,
                                         &error_message),
              "expected compiled GLOBAL hint execution success") ||
      !Expect(expect_global_hint_state(compiled_global_hint_state),
              "expected compiled GLOBAL hint state")) {
    return 1;
  }

  const std::array<std::uint32_t, 17> global_load_program_words{
      global_load_u8_words[0],   global_load_u8_words[1],
      global_load_i8_words[0],   global_load_i8_words[1],
      global_load_u16_words[0],  global_load_u16_words[1],
      global_load_i16_words[0],  global_load_i16_words[1],
      global_load_b32_words[0],  global_load_b32_words[1],
      global_load_b64_words[0],  global_load_b64_words[1],
      global_load_b96_words[0],  global_load_b96_words[1],
      global_load_b128_words[0], global_load_b128_words[1],
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> global_load_program;
  if (!Expect(decoder.DecodeProgram(global_load_program_words,
                                    &global_load_program, &error_message),
              "expected GLOBAL load program decode success") ||
      !Expect(global_load_program.size() == 9u,
              "expected nine decoded GLOBAL load instructions") ||
      !Expect(global_load_program[0].opcode == "GLOBAL_LOAD_U8",
              "expected decoded GLOBAL_LOAD_U8 opcode") ||
      !Expect(global_load_program[1].opcode == "GLOBAL_LOAD_I8",
              "expected decoded GLOBAL_LOAD_I8 opcode") ||
      !Expect(global_load_program[2].opcode == "GLOBAL_LOAD_U16",
              "expected decoded GLOBAL_LOAD_U16 opcode") ||
      !Expect(global_load_program[3].opcode == "GLOBAL_LOAD_I16",
              "expected decoded GLOBAL_LOAD_I16 opcode") ||
      !Expect(global_load_program[4].opcode == "GLOBAL_LOAD_B32",
              "expected decoded GLOBAL_LOAD_B32 opcode") ||
      !Expect(global_load_program[5].opcode == "GLOBAL_LOAD_B64",
              "expected decoded GLOBAL_LOAD_B64 opcode") ||
      !Expect(global_load_program[6].opcode == "GLOBAL_LOAD_B96",
              "expected decoded GLOBAL_LOAD_B96 opcode") ||
      !Expect(global_load_program[7].opcode == "GLOBAL_LOAD_B128",
              "expected decoded GLOBAL_LOAD_B128 opcode") ||
      !Expect(global_load_program[8].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after GLOBAL loads")) {
    return 1;
  }

  auto initialize_global_load_state = [](WaveExecutionState* state) {
    state->exec_mask = 0x80000005ull;
    state->sgprs[20] = 0x3000u;
    for (std::size_t lane = 0; lane < 32u; ++lane) {
      state->vgprs[40][lane] = static_cast<std::uint32_t>(lane * 16u);
      for (std::uint16_t reg = 20u; reg <= 33u; ++reg) {
        state->vgprs[reg][lane] = 0xdead0000u + reg;
      }
    }
  };
  auto expect_global_load_state = [](const WaveExecutionState& state) {
    if (!(state.lane_count == 32u && state.exec_mask == 0x80000005ull &&
          state.sgprs[20] == 0x3000u && state.halted &&
          !state.waiting_on_barrier && state.pc == 8u)) {
      return false;
    }

    constexpr std::uint64_t kActiveMask = 0x80000005ull;
    for (std::size_t lane = 0; lane < 32u; ++lane) {
      const bool active = (kActiveMask & (1ull << lane)) != 0u;
      if (!active) {
        for (std::uint16_t reg = 20u; reg <= 33u; ++reg) {
          if (state.vgprs[reg][lane] != 0xdead0000u + reg) {
            return false;
          }
        }
        continue;
      }

      if (state.vgprs[20][lane] !=
              static_cast<std::uint32_t>(0x40u + lane) ||
          state.vgprs[21][lane] !=
              static_cast<std::uint32_t>(
                  static_cast<std::int32_t>(static_cast<std::int8_t>(
                      static_cast<std::uint8_t>(0x80u + lane)))) ||
          state.vgprs[22][lane] !=
              static_cast<std::uint32_t>(0x8100u + lane) ||
          state.vgprs[23][lane] !=
              static_cast<std::uint32_t>(
                  static_cast<std::int32_t>(static_cast<std::int16_t>(
                      static_cast<std::uint16_t>(0x8000u + lane)))) ||
          state.vgprs[24][lane] != static_cast<std::uint32_t>(0x10000000u + lane) ||
          state.vgprs[25][lane] != static_cast<std::uint32_t>(0x20000000u + lane) ||
          state.vgprs[26][lane] != static_cast<std::uint32_t>(0x30000000u + lane) ||
          state.vgprs[27][lane] != static_cast<std::uint32_t>(0x40000000u + lane) ||
          state.vgprs[28][lane] != static_cast<std::uint32_t>(0x50000000u + lane) ||
          state.vgprs[29][lane] != static_cast<std::uint32_t>(0x60000000u + lane) ||
          state.vgprs[30][lane] != static_cast<std::uint32_t>(0x70000000u + lane) ||
          state.vgprs[31][lane] != static_cast<std::uint32_t>(0x71000000u + lane) ||
          state.vgprs[32][lane] != static_cast<std::uint32_t>(0x72000000u + lane) ||
          state.vgprs[33][lane] != static_cast<std::uint32_t>(0x73000000u + lane)) {
        return false;
      }
    }
    return true;
  };
  LinearExecutionMemory global_load_memory(0x1000u, 0x3000u);
  for (std::uint32_t lane = 0; lane < 32u; ++lane) {
    const std::uint32_t lane_address = 0x3000u + lane * 16u;
    if (!Expect(global_load_memory.StoreU8(
                    lane_address + 0x000u,
                    static_cast<std::uint8_t>(0x40u + lane)),
                "expected GLOBAL load test write for GLOBAL_LOAD_U8") ||
        !Expect(global_load_memory.StoreU8(
                    lane_address + 0x200u,
                    static_cast<std::uint8_t>(0x80u + lane)),
                "expected GLOBAL load test write for GLOBAL_LOAD_I8") ||
        !Expect(global_load_memory.StoreU16(
                    lane_address + 0x400u,
                    static_cast<std::uint16_t>(0x8100u + lane)),
                "expected GLOBAL load test write for GLOBAL_LOAD_U16") ||
        !Expect(global_load_memory.StoreU16(
                    lane_address + 0x600u,
                    static_cast<std::uint16_t>(0x8000u + lane)),
                "expected GLOBAL load test write for GLOBAL_LOAD_I16") ||
        !Expect(global_load_memory.WriteU32(
                    lane_address + 0x800u,
                    0x10000000u + lane),
                "expected GLOBAL load test write for GLOBAL_LOAD_B32") ||
        !Expect(global_load_memory.WriteU32(
                    lane_address + 0xa00u,
                    0x20000000u + lane),
                "expected GLOBAL load test write for GLOBAL_LOAD_B64 low") ||
        !Expect(global_load_memory.WriteU32(
                    lane_address + 0xa04u,
                    0x30000000u + lane),
                "expected GLOBAL load test write for GLOBAL_LOAD_B64 high") ||
        !Expect(global_load_memory.WriteU32(
                    lane_address + 0xc00u,
                    0x40000000u + lane),
                "expected GLOBAL load test write for GLOBAL_LOAD_B96 word0") ||
        !Expect(global_load_memory.WriteU32(
                    lane_address + 0xc04u,
                    0x50000000u + lane),
                "expected GLOBAL load test write for GLOBAL_LOAD_B96 word1") ||
        !Expect(global_load_memory.WriteU32(
                    lane_address + 0xc08u,
                    0x60000000u + lane),
                "expected GLOBAL load test write for GLOBAL_LOAD_B96 word2") ||
        !Expect(global_load_memory.WriteU32(
                    lane_address + 0xe00u,
                    0x70000000u + lane),
                "expected GLOBAL load test write for GLOBAL_LOAD_B128 word0") ||
        !Expect(global_load_memory.WriteU32(
                    lane_address + 0xe04u,
                    0x71000000u + lane),
                "expected GLOBAL load test write for GLOBAL_LOAD_B128 word1") ||
        !Expect(global_load_memory.WriteU32(
                    lane_address + 0xe08u,
                    0x72000000u + lane),
                "expected GLOBAL load test write for GLOBAL_LOAD_B128 word2") ||
        !Expect(global_load_memory.WriteU32(
                    lane_address + 0xe0cu,
                    0x73000000u + lane),
                "expected GLOBAL load test write for GLOBAL_LOAD_B128 word3")) {
      return 1;
    }
  }

  WaveExecutionState decoded_global_load_state;
  initialize_global_load_state(&decoded_global_load_state);
  if (!Expect(interpreter.ExecuteProgram(global_load_program,
                                         &decoded_global_load_state,
                                         &global_load_memory, &error_message),
              "expected decoded GLOBAL load execution success") ||
      !Expect(expect_global_load_state(decoded_global_load_state),
              "expected decoded GLOBAL load state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_global_load_program;
  if (!Expect(interpreter.CompileProgram(global_load_program,
                                         &compiled_global_load_program,
                                         &error_message),
              "expected compiled GLOBAL load program success") ||
      !Expect(compiled_global_load_program.size() == 9u,
              "expected nine compiled GLOBAL load instructions") ||
      !Expect(compiled_global_load_program[0].opcode ==
                  Gfx1201CompiledOpcode::kGlobalLoadU8,
              "expected compiled GLOBAL_LOAD_U8 opcode") ||
      !Expect(compiled_global_load_program[1].opcode ==
                  Gfx1201CompiledOpcode::kGlobalLoadI8,
              "expected compiled GLOBAL_LOAD_I8 opcode") ||
      !Expect(compiled_global_load_program[2].opcode ==
                  Gfx1201CompiledOpcode::kGlobalLoadU16,
              "expected compiled GLOBAL_LOAD_U16 opcode") ||
      !Expect(compiled_global_load_program[3].opcode ==
                  Gfx1201CompiledOpcode::kGlobalLoadI16,
              "expected compiled GLOBAL_LOAD_I16 opcode") ||
      !Expect(compiled_global_load_program[4].opcode ==
                  Gfx1201CompiledOpcode::kGlobalLoadB32,
              "expected compiled GLOBAL_LOAD_B32 opcode") ||
      !Expect(compiled_global_load_program[5].opcode ==
                  Gfx1201CompiledOpcode::kGlobalLoadB64,
              "expected compiled GLOBAL_LOAD_B64 opcode") ||
      !Expect(compiled_global_load_program[6].opcode ==
                  Gfx1201CompiledOpcode::kGlobalLoadB96,
              "expected compiled GLOBAL_LOAD_B96 opcode") ||
      !Expect(compiled_global_load_program[7].opcode ==
                  Gfx1201CompiledOpcode::kGlobalLoadB128,
              "expected compiled GLOBAL_LOAD_B128 opcode") ||
      !Expect(compiled_global_load_program[8].opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after GLOBAL loads")) {
    return 1;
  }

  WaveExecutionState compiled_global_load_state;
  initialize_global_load_state(&compiled_global_load_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_global_load_program,
                                         &compiled_global_load_state,
                                         &global_load_memory, &error_message),
              "expected compiled GLOBAL load execution success") ||
      !Expect(expect_global_load_state(compiled_global_load_state),
              "expected compiled GLOBAL load state")) {
    return 1;
  }

  const auto global_load_d16_u8_program_words =
      MakeGlobal(30u, 20u, 40u, 0u, 20u, 0x000);
  const auto global_load_d16_i8_program_words =
      MakeGlobal(31u, 21u, 40u, 0u, 20u, 0x200);
  const auto global_load_d16_b16_program_words =
      MakeGlobal(32u, 22u, 40u, 0u, 20u, 0x400);
  const auto global_load_d16_hi_u8_program_words =
      MakeGlobal(33u, 23u, 40u, 0u, 20u, 0x600);
  const auto global_load_d16_hi_i8_program_words =
      MakeGlobal(34u, 24u, 40u, 0u, 20u, 0x800);
  const auto global_load_d16_hi_b16_program_words =
      MakeGlobal(35u, 25u, 40u, 0u, 20u, 0xa00);
  const std::array<std::uint32_t, 13> global_load_d16_program_words{
      global_load_d16_u8_program_words[0],
      global_load_d16_u8_program_words[1],
      global_load_d16_i8_program_words[0],
      global_load_d16_i8_program_words[1],
      global_load_d16_b16_program_words[0],
      global_load_d16_b16_program_words[1],
      global_load_d16_hi_u8_program_words[0],
      global_load_d16_hi_u8_program_words[1],
      global_load_d16_hi_i8_program_words[0],
      global_load_d16_hi_i8_program_words[1],
      global_load_d16_hi_b16_program_words[0],
      global_load_d16_hi_b16_program_words[1],
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> global_load_d16_program;
  if (!Expect(decoder.DecodeProgram(global_load_d16_program_words,
                                    &global_load_d16_program, &error_message),
              "expected GLOBAL D16 load program decode success") ||
      !Expect(global_load_d16_program.size() == 7u,
              "expected seven decoded GLOBAL D16 load instructions") ||
      !Expect(global_load_d16_program[0].opcode == "GLOBAL_LOAD_D16_U8",
              "expected decoded GLOBAL_LOAD_D16_U8 opcode") ||
      !Expect(global_load_d16_program[1].opcode == "GLOBAL_LOAD_D16_I8",
              "expected decoded GLOBAL_LOAD_D16_I8 opcode") ||
      !Expect(global_load_d16_program[2].opcode == "GLOBAL_LOAD_D16_B16",
              "expected decoded GLOBAL_LOAD_D16_B16 opcode") ||
      !Expect(global_load_d16_program[3].opcode == "GLOBAL_LOAD_D16_HI_U8",
              "expected decoded GLOBAL_LOAD_D16_HI_U8 opcode") ||
      !Expect(global_load_d16_program[4].opcode == "GLOBAL_LOAD_D16_HI_I8",
              "expected decoded GLOBAL_LOAD_D16_HI_I8 opcode") ||
      !Expect(global_load_d16_program[5].opcode == "GLOBAL_LOAD_D16_HI_B16",
              "expected decoded GLOBAL_LOAD_D16_HI_B16 opcode") ||
      !Expect(global_load_d16_program[6].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after GLOBAL D16 loads")) {
    return 1;
  }

  auto initialize_global_load_d16_state = [](WaveExecutionState* state) {
    state->exec_mask = 0x80000005ull;
    state->sgprs[20] = 0x5000u;
    for (std::size_t lane = 0; lane < 32u; ++lane) {
      state->vgprs[40][lane] = static_cast<std::uint32_t>(lane * 16u);
      state->vgprs[20][lane] = 0xa10000a0u + static_cast<std::uint32_t>(lane);
      state->vgprs[21][lane] = 0xa20000b0u + static_cast<std::uint32_t>(lane);
      state->vgprs[22][lane] = 0xa30000c0u + static_cast<std::uint32_t>(lane);
      state->vgprs[23][lane] = 0xd4000100u + static_cast<std::uint32_t>(lane);
      state->vgprs[24][lane] = 0xd5000200u + static_cast<std::uint32_t>(lane);
      state->vgprs[25][lane] = 0xd6000300u + static_cast<std::uint32_t>(lane);
    }
  };
  auto expect_global_load_d16_state = [](const WaveExecutionState& state) {
    if (!(state.lane_count == 32u && state.exec_mask == 0x80000005ull &&
          state.sgprs[20] == 0x5000u && state.halted &&
          !state.waiting_on_barrier && state.pc == 6u)) {
      return false;
    }

    constexpr std::uint64_t kActiveMask = 0x80000005ull;
    for (std::size_t lane = 0; lane < 32u; ++lane) {
      const bool active = (kActiveMask & (1ull << lane)) != 0u;
      const std::uint32_t initial_u8 = 0xa10000a0u + static_cast<std::uint32_t>(lane);
      const std::uint32_t initial_i8 = 0xa20000b0u + static_cast<std::uint32_t>(lane);
      const std::uint32_t initial_b16 =
          0xa30000c0u + static_cast<std::uint32_t>(lane);
      const std::uint32_t initial_hi_u8 =
          0xd4000100u + static_cast<std::uint32_t>(lane);
      const std::uint32_t initial_hi_i8 =
          0xd5000200u + static_cast<std::uint32_t>(lane);
      const std::uint32_t initial_hi_b16 =
          0xd6000300u + static_cast<std::uint32_t>(lane);

      const std::uint32_t expected_u8 =
          active ? ((initial_u8 & 0xffff0000u) |
                    static_cast<std::uint16_t>(0x40u + lane))
                 : initial_u8;
      const std::uint32_t expected_i8 =
          active ? ((initial_i8 & 0xffff0000u) |
                    static_cast<std::uint16_t>(
                        static_cast<std::int16_t>(static_cast<std::int8_t>(
                            static_cast<std::uint8_t>(0x80u + lane)))))
                 : initial_i8;
      const std::uint32_t expected_b16 =
          active ? ((initial_b16 & 0xffff0000u) |
                    static_cast<std::uint16_t>(0x9100u + lane))
                 : initial_b16;
      const std::uint16_t hi_u8_payload =
          static_cast<std::uint16_t>(0x50u + lane);
      const std::uint16_t hi_i8_payload = static_cast<std::uint16_t>(
          static_cast<std::int16_t>(static_cast<std::int8_t>(
              static_cast<std::uint8_t>(0x90u + lane))));
      const std::uint16_t hi_b16_payload =
          static_cast<std::uint16_t>(0xa100u + lane);
      const std::uint32_t expected_hi_u8 =
          active ? ((initial_hi_u8 & 0x0000ffffu) |
                    (static_cast<std::uint32_t>(hi_u8_payload) << 16))
                 : initial_hi_u8;
      const std::uint32_t expected_hi_i8 =
          active ? ((initial_hi_i8 & 0x0000ffffu) |
                    (static_cast<std::uint32_t>(hi_i8_payload) << 16))
                 : initial_hi_i8;
      const std::uint32_t expected_hi_b16 =
          active ? ((initial_hi_b16 & 0x0000ffffu) |
                    (static_cast<std::uint32_t>(hi_b16_payload) << 16))
                 : initial_hi_b16;

      if (state.vgprs[40][lane] != static_cast<std::uint32_t>(lane * 16u) ||
          state.vgprs[20][lane] != expected_u8 ||
          state.vgprs[21][lane] != expected_i8 ||
          state.vgprs[22][lane] != expected_b16 ||
          state.vgprs[23][lane] != expected_hi_u8 ||
          state.vgprs[24][lane] != expected_hi_i8 ||
          state.vgprs[25][lane] != expected_hi_b16) {
        return false;
      }
    }
    return true;
  };
  LinearExecutionMemory global_load_d16_memory(0x1000u, 0x5000u);
  for (std::uint32_t lane = 0; lane < 32u; ++lane) {
    const std::uint32_t lane_address = 0x5000u + lane * 16u;
    if (!Expect(global_load_d16_memory.StoreU8(
                    lane_address + 0x000u,
                    static_cast<std::uint8_t>(0x40u + lane)),
                "expected GLOBAL D16 load test write for GLOBAL_LOAD_D16_U8") ||
        !Expect(global_load_d16_memory.StoreU8(
                    lane_address + 0x200u,
                    static_cast<std::uint8_t>(0x80u + lane)),
                "expected GLOBAL D16 load test write for GLOBAL_LOAD_D16_I8") ||
        !Expect(global_load_d16_memory.StoreU16(
                    lane_address + 0x400u,
                    static_cast<std::uint16_t>(0x9100u + lane)),
                "expected GLOBAL D16 load test write for GLOBAL_LOAD_D16_B16") ||
        !Expect(global_load_d16_memory.StoreU8(
                    lane_address + 0x600u,
                    static_cast<std::uint8_t>(0x50u + lane)),
                "expected GLOBAL D16 load test write for GLOBAL_LOAD_D16_HI_U8") ||
        !Expect(global_load_d16_memory.StoreU8(
                    lane_address + 0x800u,
                    static_cast<std::uint8_t>(0x90u + lane)),
                "expected GLOBAL D16 load test write for GLOBAL_LOAD_D16_HI_I8") ||
        !Expect(global_load_d16_memory.StoreU16(
                    lane_address + 0xa00u,
                    static_cast<std::uint16_t>(0xa100u + lane)),
                "expected GLOBAL D16 load test write for GLOBAL_LOAD_D16_HI_B16")) {
      return 1;
    }
  }

  WaveExecutionState decoded_global_load_d16_state;
  initialize_global_load_d16_state(&decoded_global_load_d16_state);
  if (!Expect(interpreter.ExecuteProgram(global_load_d16_program,
                                         &decoded_global_load_d16_state,
                                         &global_load_d16_memory,
                                         &error_message),
              "expected decoded GLOBAL D16 load execution success") ||
      !Expect(expect_global_load_d16_state(decoded_global_load_d16_state),
              "expected decoded GLOBAL D16 load state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_global_load_d16_program;
  if (!Expect(interpreter.CompileProgram(global_load_d16_program,
                                         &compiled_global_load_d16_program,
                                         &error_message),
              "expected compiled GLOBAL D16 load program success") ||
      !Expect(compiled_global_load_d16_program.size() == 7u,
              "expected seven compiled GLOBAL D16 load instructions") ||
      !Expect(compiled_global_load_d16_program[0].opcode ==
                  Gfx1201CompiledOpcode::kGlobalLoadD16U8,
              "expected compiled GLOBAL_LOAD_D16_U8 opcode") ||
      !Expect(compiled_global_load_d16_program[1].opcode ==
                  Gfx1201CompiledOpcode::kGlobalLoadD16I8,
              "expected compiled GLOBAL_LOAD_D16_I8 opcode") ||
      !Expect(compiled_global_load_d16_program[2].opcode ==
                  Gfx1201CompiledOpcode::kGlobalLoadD16B16,
              "expected compiled GLOBAL_LOAD_D16_B16 opcode") ||
      !Expect(compiled_global_load_d16_program[3].opcode ==
                  Gfx1201CompiledOpcode::kGlobalLoadD16HiU8,
              "expected compiled GLOBAL_LOAD_D16_HI_U8 opcode") ||
      !Expect(compiled_global_load_d16_program[4].opcode ==
                  Gfx1201CompiledOpcode::kGlobalLoadD16HiI8,
              "expected compiled GLOBAL_LOAD_D16_HI_I8 opcode") ||
      !Expect(compiled_global_load_d16_program[5].opcode ==
                  Gfx1201CompiledOpcode::kGlobalLoadD16HiB16,
              "expected compiled GLOBAL_LOAD_D16_HI_B16 opcode") ||
      !Expect(compiled_global_load_d16_program[6].opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after GLOBAL D16 loads")) {
    return 1;
  }

  WaveExecutionState compiled_global_load_d16_state;
  initialize_global_load_d16_state(&compiled_global_load_d16_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_global_load_d16_program,
                                         &compiled_global_load_d16_state,
                                         &global_load_d16_memory,
                                         &error_message),
              "expected compiled GLOBAL D16 load execution success") ||
      !Expect(expect_global_load_d16_state(compiled_global_load_d16_state),
              "expected compiled GLOBAL D16 load state")) {
    return 1;
  }

  const auto global_load_tr_b64_program_words =
      MakeGlobal(88u, 20u, 40u, 0u, 20u, 0x100);
  const auto global_load_tr_b128_program_words =
      MakeGlobal(87u, 22u, 40u, 0u, 20u, 0x300);
  const std::array<std::uint32_t, 5> global_load_tr_program_words{
      global_load_tr_b64_program_words[0],
      global_load_tr_b64_program_words[1],
      global_load_tr_b128_program_words[0],
      global_load_tr_b128_program_words[1],
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> global_load_tr_program;
  if (!Expect(decoder.DecodeProgram(global_load_tr_program_words,
                                    &global_load_tr_program, &error_message),
              "expected GLOBAL TR load program decode success") ||
      !Expect(global_load_tr_program.size() == 3u,
              "expected three decoded GLOBAL TR load instructions") ||
      !Expect(global_load_tr_program[0].opcode == "GLOBAL_LOAD_TR_B64",
              "expected decoded GLOBAL_LOAD_TR_B64 opcode") ||
      !Expect(global_load_tr_program[1].opcode == "GLOBAL_LOAD_TR_B128",
              "expected decoded GLOBAL_LOAD_TR_B128 opcode") ||
      !Expect(global_load_tr_program[2].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after GLOBAL TR loads")) {
    return 1;
  }

  auto initialize_global_load_tr_state = [](WaveExecutionState* state) {
    state->exec_mask = 0x80000005ull;
    state->sgprs[20] = 0x7000u;
    for (std::size_t lane = 0; lane < 32u; ++lane) {
      state->vgprs[40][lane] = static_cast<std::uint32_t>(lane * 16u);
      state->vgprs[20][lane] = 0xc10000a0u + static_cast<std::uint32_t>(lane);
      state->vgprs[21][lane] = 0xc20000b0u + static_cast<std::uint32_t>(lane);
      state->vgprs[22][lane] = 0xd30000c0u + static_cast<std::uint32_t>(lane);
      state->vgprs[23][lane] = 0xd40000d0u + static_cast<std::uint32_t>(lane);
      state->vgprs[24][lane] = 0xd50000e0u + static_cast<std::uint32_t>(lane);
      state->vgprs[25][lane] = 0xd60000f0u + static_cast<std::uint32_t>(lane);
    }
  };
  auto expect_global_load_tr_state = [](const WaveExecutionState& state) {
    if (!(state.lane_count == 32u && state.exec_mask == 0x80000005ull &&
          state.sgprs[20] == 0x7000u && state.halted &&
          !state.waiting_on_barrier && state.pc == 2u)) {
      return false;
    }

    constexpr std::uint64_t kActiveMask = 0x80000005ull;
    for (std::size_t lane = 0; lane < 32u; ++lane) {
      const bool active = (kActiveMask & (1ull << lane)) != 0u;
      const std::uint32_t initial_b64_lo =
          0xc10000a0u + static_cast<std::uint32_t>(lane);
      const std::uint32_t initial_b64_hi =
          0xc20000b0u + static_cast<std::uint32_t>(lane);
      const std::uint32_t initial_b128_0 =
          0xd30000c0u + static_cast<std::uint32_t>(lane);
      const std::uint32_t initial_b128_1 =
          0xd40000d0u + static_cast<std::uint32_t>(lane);
      const std::uint32_t initial_b128_2 =
          0xd50000e0u + static_cast<std::uint32_t>(lane);
      const std::uint32_t initial_b128_3 =
          0xd60000f0u + static_cast<std::uint32_t>(lane);
      if (state.vgprs[40][lane] != static_cast<std::uint32_t>(lane * 16u) ||
          state.vgprs[20][lane] !=
              (active ? (0x81000000u + static_cast<std::uint32_t>(lane))
                      : initial_b64_lo) ||
          state.vgprs[21][lane] !=
              (active ? (0x82000000u + static_cast<std::uint32_t>(lane))
                      : initial_b64_hi) ||
          state.vgprs[22][lane] !=
              (active ? (0x91000000u + static_cast<std::uint32_t>(lane))
                      : initial_b128_0) ||
          state.vgprs[23][lane] !=
              (active ? (0x92000000u + static_cast<std::uint32_t>(lane))
                      : initial_b128_1) ||
          state.vgprs[24][lane] !=
              (active ? (0x93000000u + static_cast<std::uint32_t>(lane))
                      : initial_b128_2) ||
          state.vgprs[25][lane] !=
              (active ? (0x94000000u + static_cast<std::uint32_t>(lane))
                      : initial_b128_3)) {
        return false;
      }
    }
    return true;
  };
  LinearExecutionMemory global_load_tr_memory(0x1000u, 0x7000u);
  for (std::uint32_t lane = 0; lane < 32u; ++lane) {
    const std::uint32_t lane_address = 0x7000u + lane * 16u;
    if (!Expect(global_load_tr_memory.StoreU32(
                    lane_address + 0x100u, 0x81000000u + lane),
                "expected GLOBAL TR load test write for GLOBAL_LOAD_TR_B64 low") ||
        !Expect(global_load_tr_memory.StoreU32(
                    lane_address + 0x104u, 0x82000000u + lane),
                "expected GLOBAL TR load test write for GLOBAL_LOAD_TR_B64 high") ||
        !Expect(global_load_tr_memory.StoreU32(
                    lane_address + 0x300u, 0x91000000u + lane),
                "expected GLOBAL TR load test write for GLOBAL_LOAD_TR_B128 word0") ||
        !Expect(global_load_tr_memory.StoreU32(
                    lane_address + 0x304u, 0x92000000u + lane),
                "expected GLOBAL TR load test write for GLOBAL_LOAD_TR_B128 word1") ||
        !Expect(global_load_tr_memory.StoreU32(
                    lane_address + 0x308u, 0x93000000u + lane),
                "expected GLOBAL TR load test write for GLOBAL_LOAD_TR_B128 word2") ||
        !Expect(global_load_tr_memory.StoreU32(
                    lane_address + 0x30cu, 0x94000000u + lane),
                "expected GLOBAL TR load test write for GLOBAL_LOAD_TR_B128 word3")) {
      return 1;
    }
  }

  WaveExecutionState decoded_global_load_tr_state;
  initialize_global_load_tr_state(&decoded_global_load_tr_state);
  if (!Expect(interpreter.ExecuteProgram(global_load_tr_program,
                                         &decoded_global_load_tr_state,
                                         &global_load_tr_memory,
                                         &error_message),
              "expected decoded GLOBAL TR load execution success") ||
      !Expect(expect_global_load_tr_state(decoded_global_load_tr_state),
              "expected decoded GLOBAL TR load state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_global_load_tr_program;
  if (!Expect(interpreter.CompileProgram(global_load_tr_program,
                                         &compiled_global_load_tr_program,
                                         &error_message),
              "expected compiled GLOBAL TR load program success") ||
      !Expect(compiled_global_load_tr_program.size() == 3u,
              "expected three compiled GLOBAL TR load instructions") ||
      !Expect(compiled_global_load_tr_program[0].opcode ==
                  Gfx1201CompiledOpcode::kGlobalLoadTrB64,
              "expected compiled GLOBAL_LOAD_TR_B64 opcode") ||
      !Expect(compiled_global_load_tr_program[1].opcode ==
                  Gfx1201CompiledOpcode::kGlobalLoadTrB128,
              "expected compiled GLOBAL_LOAD_TR_B128 opcode") ||
      !Expect(compiled_global_load_tr_program[2].opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after GLOBAL TR loads")) {
    return 1;
  }

  WaveExecutionState compiled_global_load_tr_state;
  initialize_global_load_tr_state(&compiled_global_load_tr_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_global_load_tr_program,
                                         &compiled_global_load_tr_state,
                                         &global_load_tr_memory,
                                         &error_message),
              "expected compiled GLOBAL TR load execution success") ||
      !Expect(expect_global_load_tr_state(compiled_global_load_tr_state),
              "expected compiled GLOBAL TR load state")) {
    return 1;
  }

  const auto global_load_addtid_b32_program_words =
      MakeGlobal(40u, 20u, 0u, 0u, 20u, 0);
  const std::array<std::uint32_t, 3> global_load_addtid_program_words{
      global_load_addtid_b32_program_words[0],
      global_load_addtid_b32_program_words[1],
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> global_load_addtid_program;
  if (!Expect(decoder.DecodeProgram(global_load_addtid_program_words,
                                    &global_load_addtid_program,
                                    &error_message),
              "expected GLOBAL ADDTID load program decode success") ||
      !Expect(global_load_addtid_program.size() == 2u,
              "expected two decoded GLOBAL ADDTID load instructions") ||
      !Expect(global_load_addtid_program[0].opcode == "GLOBAL_LOAD_ADDTID_B32",
              "expected decoded GLOBAL_LOAD_ADDTID_B32 opcode") ||
      !Expect(global_load_addtid_program[1].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after GLOBAL ADDTID load")) {
    return 1;
  }

  auto initialize_global_load_addtid_state = [](WaveExecutionState* state) {
    state->exec_mask = 0x80000005ull;
    state->sgprs[20] = 0x8000u;
    state->sgprs[21] = 0u;
    for (std::size_t lane = 0; lane < 32u; ++lane) {
      state->vgprs[20][lane] = 0xe10000a0u + static_cast<std::uint32_t>(lane);
    }
  };
  auto expect_global_load_addtid_state = [](const WaveExecutionState& state) {
    if (!(state.lane_count == 32u && state.exec_mask == 0x80000005ull &&
          state.sgprs[20] == 0x8000u && state.sgprs[21] == 0u &&
          state.halted && !state.waiting_on_barrier && state.pc == 1u)) {
      return false;
    }
    constexpr std::uint64_t kActiveMask = 0x80000005ull;
    for (std::size_t lane = 0; lane < 32u; ++lane) {
      const bool active = (kActiveMask & (1ull << lane)) != 0u;
      const std::uint32_t initial_value =
          0xe10000a0u + static_cast<std::uint32_t>(lane);
      const std::uint32_t expected_value =
          active ? (0x61000000u + static_cast<std::uint32_t>(lane))
                 : initial_value;
      if (state.vgprs[20][lane] != expected_value) {
        return false;
      }
    }
    return true;
  };
  LinearExecutionMemory global_load_addtid_memory(0x1000u, 0x8000u);
  for (std::uint32_t lane = 0; lane < 32u; ++lane) {
    if (!Expect(global_load_addtid_memory.StoreU32(
                    0x8000u + lane * 4u, 0x61000000u + lane),
                "expected GLOBAL ADDTID load test write")) {
      return 1;
    }
  }

  WaveExecutionState decoded_global_load_addtid_state;
  initialize_global_load_addtid_state(&decoded_global_load_addtid_state);
  if (!Expect(interpreter.ExecuteProgram(global_load_addtid_program,
                                         &decoded_global_load_addtid_state,
                                         &global_load_addtid_memory,
                                         &error_message),
              "expected decoded GLOBAL ADDTID load execution success") ||
      !Expect(expect_global_load_addtid_state(decoded_global_load_addtid_state),
              "expected decoded GLOBAL ADDTID load state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_global_load_addtid_program;
  if (!Expect(interpreter.CompileProgram(global_load_addtid_program,
                                         &compiled_global_load_addtid_program,
                                         &error_message),
              "expected compiled GLOBAL ADDTID load program success") ||
      !Expect(compiled_global_load_addtid_program.size() == 2u,
              "expected two compiled GLOBAL ADDTID load instructions") ||
      !Expect(compiled_global_load_addtid_program[0].opcode ==
                  Gfx1201CompiledOpcode::kGlobalLoadAddtidB32,
              "expected compiled GLOBAL_LOAD_ADDTID_B32 opcode") ||
      !Expect(compiled_global_load_addtid_program[1].opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after GLOBAL ADDTID load")) {
    return 1;
  }

  WaveExecutionState compiled_global_load_addtid_state;
  initialize_global_load_addtid_state(&compiled_global_load_addtid_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_global_load_addtid_program,
                                         &compiled_global_load_addtid_state,
                                         &global_load_addtid_memory,
                                         &error_message),
              "expected compiled GLOBAL ADDTID load execution success") ||
      !Expect(expect_global_load_addtid_state(compiled_global_load_addtid_state),
              "expected compiled GLOBAL ADDTID load state")) {
    return 1;
  }

  const auto global_load_block_b32_program_words =
      MakeGlobal(83u, 80u, 40u, 0u, 20u, 16);
  const std::array<std::uint32_t, 3> global_load_block_program_words{
      global_load_block_b32_program_words[0],
      global_load_block_b32_program_words[1],
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> global_load_block_program;
  if (!Expect(decoder.DecodeProgram(global_load_block_program_words,
                                    &global_load_block_program,
                                    &error_message),
              "expected GLOBAL BLOCK load program decode success") ||
      !Expect(global_load_block_program.size() == 2u,
              "expected two decoded GLOBAL BLOCK load instructions") ||
      !Expect(global_load_block_program[0].opcode == "GLOBAL_LOAD_BLOCK",
              "expected decoded GLOBAL_LOAD_BLOCK opcode") ||
      !Expect(global_load_block_program[1].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after GLOBAL BLOCK load")) {
    return 1;
  }

  auto initialize_global_load_block_state = [](WaveExecutionState* state) {
    state->exec_mask = 0x80000005ull;
    state->sgprs[20] = 0xa000u;
    state->sgprs[21] = 0u;
    state->sgprs[kM0RegisterIndex] = (1u << 0) | (1u << 2) | (1u << 31);
    for (std::size_t lane = 0; lane < 32u; ++lane) {
      state->vgprs[40][lane] = static_cast<std::uint32_t>(lane) * 0x100u;
      for (std::size_t slot = 0; slot < 32u; ++slot) {
        state->vgprs[80u + slot][lane] =
            0xd0000000u + static_cast<std::uint32_t>(slot << 8) +
            static_cast<std::uint32_t>(lane);
      }
    }
  };
  auto expect_global_load_block_state = [](const WaveExecutionState& state) {
    if (!(state.lane_count == 32u && state.exec_mask == 0x80000005ull &&
          state.sgprs[20] == 0xa000u && state.sgprs[21] == 0u &&
          state.sgprs[kM0RegisterIndex] == ((1u << 0) | (1u << 2) | (1u << 31)) &&
          state.halted && !state.waiting_on_barrier && state.pc == 1u)) {
      return false;
    }
    constexpr std::uint64_t kActiveMask = 0x80000005ull;
    constexpr std::uint32_t kBlockMask = (1u << 0) | (1u << 2) | (1u << 31);
    for (std::size_t lane = 0; lane < 32u; ++lane) {
      for (std::size_t slot = 0; slot < 32u; ++slot) {
        const bool active = (kActiveMask & (1ull << lane)) != 0u;
        const bool selected =
            (kBlockMask & (1u << static_cast<std::uint32_t>(slot))) != 0u;
        const std::uint32_t initial_value =
            0xd0000000u + static_cast<std::uint32_t>(slot << 8) +
            static_cast<std::uint32_t>(lane);
        const std::uint32_t expected_value =
            (active && selected)
                ? (0x63000000u + static_cast<std::uint32_t>(lane << 8) +
                   static_cast<std::uint32_t>(slot))
                : initial_value;
        if (state.vgprs[80u + slot][lane] != expected_value) {
          return false;
        }
      }
    }
    return true;
  };
  LinearExecutionMemory global_load_block_memory(0x3000u, 0xa000u);
  for (std::uint32_t lane : {0u, 2u, 31u}) {
    const std::uint32_t lane_base = 0xa000u + lane * 0x100u + 16u;
    for (std::uint32_t slot : {0u, 2u, 31u}) {
      if (!Expect(global_load_block_memory.StoreU32(
                      lane_base + slot * 4u,
                      0x63000000u + (lane << 8) + slot),
                  "expected GLOBAL BLOCK load test write")) {
        return 1;
      }
    }
  }

  WaveExecutionState decoded_global_load_block_state;
  initialize_global_load_block_state(&decoded_global_load_block_state);
  if (!Expect(interpreter.ExecuteProgram(global_load_block_program,
                                         &decoded_global_load_block_state,
                                         &global_load_block_memory,
                                         &error_message),
              "expected decoded GLOBAL BLOCK load execution success") ||
      !Expect(expect_global_load_block_state(decoded_global_load_block_state),
              "expected decoded GLOBAL BLOCK load state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_global_load_block_program;
  if (!Expect(interpreter.CompileProgram(global_load_block_program,
                                         &compiled_global_load_block_program,
                                         &error_message),
              "expected compiled GLOBAL BLOCK load program success") ||
      !Expect(compiled_global_load_block_program.size() == 2u,
              "expected two compiled GLOBAL BLOCK load instructions") ||
      !Expect(compiled_global_load_block_program[0].opcode ==
                  Gfx1201CompiledOpcode::kGlobalLoadBlock,
              "expected compiled GLOBAL_LOAD_BLOCK opcode") ||
      !Expect(compiled_global_load_block_program[1].opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after GLOBAL BLOCK load")) {
    return 1;
  }

  WaveExecutionState compiled_global_load_block_state;
  initialize_global_load_block_state(&compiled_global_load_block_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_global_load_block_program,
                                         &compiled_global_load_block_state,
                                         &global_load_block_memory,
                                         &error_message),
              "expected compiled GLOBAL BLOCK load execution success") ||
      !Expect(expect_global_load_block_state(compiled_global_load_block_state),
              "expected compiled GLOBAL BLOCK load state")) {
    return 1;
  }

  const auto global_store_addtid_b32_program_words =
      MakeGlobal(41u, 0u, 0u, 20u, 22u, 0);
  const std::array<std::uint32_t, 3> global_store_addtid_program_words{
      global_store_addtid_b32_program_words[0],
      global_store_addtid_b32_program_words[1],
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> global_store_addtid_program;
  if (!Expect(decoder.DecodeProgram(global_store_addtid_program_words,
                                    &global_store_addtid_program,
                                    &error_message),
              "expected GLOBAL ADDTID store program decode success") ||
      !Expect(global_store_addtid_program.size() == 2u,
              "expected two decoded GLOBAL ADDTID store instructions") ||
      !Expect(global_store_addtid_program[0].opcode == "GLOBAL_STORE_ADDTID_B32",
              "expected decoded GLOBAL_STORE_ADDTID_B32 opcode") ||
      !Expect(global_store_addtid_program[1].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after GLOBAL ADDTID store")) {
    return 1;
  }

  auto initialize_global_store_addtid_state = [](WaveExecutionState* state) {
    state->exec_mask = 0x80000005ull;
    state->sgprs[22] = 0x9000u;
    state->sgprs[23] = 0u;
    for (std::size_t lane = 0; lane < 32u; ++lane) {
      state->vgprs[20][lane] = 0x71000000u + static_cast<std::uint32_t>(lane);
    }
  };
  auto expect_global_store_addtid_state = [](const WaveExecutionState& state) {
    if (!(state.lane_count == 32u && state.exec_mask == 0x80000005ull &&
          state.sgprs[22] == 0x9000u && state.sgprs[23] == 0u &&
          state.halted && !state.waiting_on_barrier && state.pc == 1u)) {
      return false;
    }
    for (std::size_t lane = 0; lane < 32u; ++lane) {
      if (state.vgprs[20][lane] !=
          0x71000000u + static_cast<std::uint32_t>(lane)) {
        return false;
      }
    }
    return true;
  };
  auto expect_global_store_addtid_memory = [](LinearExecutionMemory* memory) {
    constexpr std::uint64_t kActiveMask = 0x80000005ull;
    for (std::uint32_t lane = 0; lane < 32u; ++lane) {
      std::uint32_t value = 0;
      if (!memory->LoadU32(0x9000u + lane * 4u, &value)) {
        return false;
      }
      const bool active = (kActiveMask & (1ull << lane)) != 0u;
      const std::uint32_t expected =
          active ? (0x71000000u + lane) : 0u;
      if (value != expected) {
        return false;
      }
    }
    return true;
  };

  LinearExecutionMemory decoded_global_store_addtid_memory(0x1000u, 0x9000u);
  WaveExecutionState decoded_global_store_addtid_state;
  initialize_global_store_addtid_state(&decoded_global_store_addtid_state);
  if (!Expect(interpreter.ExecuteProgram(global_store_addtid_program,
                                         &decoded_global_store_addtid_state,
                                         &decoded_global_store_addtid_memory,
                                         &error_message),
              "expected decoded GLOBAL ADDTID store execution success") ||
      !Expect(expect_global_store_addtid_state(decoded_global_store_addtid_state),
              "expected decoded GLOBAL ADDTID store state") ||
      !Expect(expect_global_store_addtid_memory(
                  &decoded_global_store_addtid_memory),
              "expected decoded GLOBAL ADDTID store memory state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_global_store_addtid_program;
  if (!Expect(interpreter.CompileProgram(global_store_addtid_program,
                                         &compiled_global_store_addtid_program,
                                         &error_message),
              "expected compiled GLOBAL ADDTID store program success") ||
      !Expect(compiled_global_store_addtid_program.size() == 2u,
              "expected two compiled GLOBAL ADDTID store instructions") ||
      !Expect(compiled_global_store_addtid_program[0].opcode ==
                  Gfx1201CompiledOpcode::kGlobalStoreAddtidB32,
              "expected compiled GLOBAL_STORE_ADDTID_B32 opcode") ||
      !Expect(compiled_global_store_addtid_program[1].opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after GLOBAL ADDTID store")) {
    return 1;
  }

  LinearExecutionMemory compiled_global_store_addtid_memory(0x1000u, 0x9000u);
  WaveExecutionState compiled_global_store_addtid_state;
  initialize_global_store_addtid_state(&compiled_global_store_addtid_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_global_store_addtid_program,
                                         &compiled_global_store_addtid_state,
                                         &compiled_global_store_addtid_memory,
                                         &error_message),
              "expected compiled GLOBAL ADDTID store execution success") ||
      !Expect(expect_global_store_addtid_state(compiled_global_store_addtid_state),
              "expected compiled GLOBAL ADDTID store state") ||
      !Expect(expect_global_store_addtid_memory(
                  &compiled_global_store_addtid_memory),
              "expected compiled GLOBAL ADDTID store memory state")) {
    return 1;
  }

  const auto global_store_block_program_words_raw =
      MakeGlobal(84u, 0u, 41u, 90u, 22u, 12);
  const std::array<std::uint32_t, 3> global_store_block_program_words{
      global_store_block_program_words_raw[0],
      global_store_block_program_words_raw[1],
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> global_store_block_program;
  if (!Expect(decoder.DecodeProgram(global_store_block_program_words,
                                    &global_store_block_program,
                                    &error_message),
              "expected GLOBAL BLOCK store program decode success") ||
      !Expect(global_store_block_program.size() == 2u,
              "expected two decoded GLOBAL BLOCK store instructions") ||
      !Expect(global_store_block_program[0].opcode == "GLOBAL_STORE_BLOCK",
              "expected decoded GLOBAL_STORE_BLOCK opcode") ||
      !Expect(global_store_block_program[1].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after GLOBAL BLOCK store")) {
    return 1;
  }

  auto initialize_global_store_block_state = [](WaveExecutionState* state) {
    state->exec_mask = 0x80000005ull;
    state->sgprs[22] = 0xb000u;
    state->sgprs[23] = 0u;
    state->sgprs[kM0RegisterIndex] = (1u << 1) | (1u << 3) | (1u << 31);
    for (std::size_t lane = 0; lane < 32u; ++lane) {
      state->vgprs[41][lane] = static_cast<std::uint32_t>(lane) * 0x100u;
      for (std::size_t slot = 0; slot < 32u; ++slot) {
        state->vgprs[90u + slot][lane] =
            0x74000000u + static_cast<std::uint32_t>(slot << 8) +
            static_cast<std::uint32_t>(lane);
      }
    }
  };
  auto expect_global_store_block_state = [](const WaveExecutionState& state) {
    if (!(state.lane_count == 32u && state.exec_mask == 0x80000005ull &&
          state.sgprs[22] == 0xb000u && state.sgprs[23] == 0u &&
          state.sgprs[kM0RegisterIndex] == ((1u << 1) | (1u << 3) | (1u << 31)) &&
          state.halted && !state.waiting_on_barrier && state.pc == 1u)) {
      return false;
    }
    for (std::size_t lane = 0; lane < 32u; ++lane) {
      if (state.vgprs[41][lane] != static_cast<std::uint32_t>(lane) * 0x100u) {
        return false;
      }
      for (std::size_t slot = 0; slot < 32u; ++slot) {
        const std::uint32_t expected_value =
            0x74000000u + static_cast<std::uint32_t>(slot << 8) +
            static_cast<std::uint32_t>(lane);
        if (state.vgprs[90u + slot][lane] != expected_value) {
          return false;
        }
      }
    }
    return true;
  };
  auto expect_global_store_block_memory = [](LinearExecutionMemory* memory) {
    constexpr std::uint64_t kActiveMask = 0x80000005ull;
    constexpr std::uint32_t kBlockMask = (1u << 1) | (1u << 3) | (1u << 31);
    for (std::uint32_t lane = 0; lane < 32u; ++lane) {
      const std::uint32_t lane_base = 0xb000u + lane * 0x100u + 12u;
      for (std::uint32_t slot = 0; slot < 32u; ++slot) {
        std::uint32_t value = 0;
        if (!memory->LoadU32(lane_base + slot * 4u, &value)) {
          return false;
        }
        const bool active = (kActiveMask & (1ull << lane)) != 0u;
        const bool selected =
            (kBlockMask & (1u << static_cast<std::uint32_t>(slot))) != 0u;
        const std::uint32_t expected_value =
            (active && selected)
                ? (0x74000000u + (slot << 8) + lane)
                : 0u;
        if (value != expected_value) {
          return false;
        }
      }
    }
    return true;
  };

  LinearExecutionMemory decoded_global_store_block_memory(0x3000u, 0xb000u);
  WaveExecutionState decoded_global_store_block_state;
  initialize_global_store_block_state(&decoded_global_store_block_state);
  if (!Expect(interpreter.ExecuteProgram(global_store_block_program,
                                         &decoded_global_store_block_state,
                                         &decoded_global_store_block_memory,
                                         &error_message),
              "expected decoded GLOBAL BLOCK store execution success") ||
      !Expect(expect_global_store_block_state(decoded_global_store_block_state),
              "expected decoded GLOBAL BLOCK store state") ||
      !Expect(expect_global_store_block_memory(
                  &decoded_global_store_block_memory),
              "expected decoded GLOBAL BLOCK store memory state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_global_store_block_program;
  if (!Expect(interpreter.CompileProgram(global_store_block_program,
                                         &compiled_global_store_block_program,
                                         &error_message),
              "expected compiled GLOBAL BLOCK store program success") ||
      !Expect(compiled_global_store_block_program.size() == 2u,
              "expected two compiled GLOBAL BLOCK store instructions") ||
      !Expect(compiled_global_store_block_program[0].opcode ==
                  Gfx1201CompiledOpcode::kGlobalStoreBlock,
              "expected compiled GLOBAL_STORE_BLOCK opcode") ||
      !Expect(compiled_global_store_block_program[1].opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after GLOBAL BLOCK store")) {
    return 1;
  }

  LinearExecutionMemory compiled_global_store_block_memory(0x3000u, 0xb000u);
  WaveExecutionState compiled_global_store_block_state;
  initialize_global_store_block_state(&compiled_global_store_block_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_global_store_block_program,
                                         &compiled_global_store_block_state,
                                         &compiled_global_store_block_memory,
                                         &error_message),
              "expected compiled GLOBAL BLOCK store execution success") ||
      !Expect(expect_global_store_block_state(compiled_global_store_block_state),
              "expected compiled GLOBAL BLOCK store state") ||
      !Expect(expect_global_store_block_memory(
                  &compiled_global_store_block_memory),
              "expected compiled GLOBAL BLOCK store memory state")) {
    return 1;
  }

  const auto global_store_b8_words = MakeGlobal(24u, 0u, 40u, 20u, 20u, 0x000);
  const auto global_store_b16_words =
      MakeGlobal(25u, 0u, 40u, 21u, 20u, 0x200);
  const auto global_store_b32_program_words =
      MakeGlobal(26u, 0u, 40u, 22u, 20u, 0x400);
  const auto global_store_b64_words =
      MakeGlobal(27u, 0u, 40u, 23u, 20u, 0x600);
  const auto global_store_b96_words =
      MakeGlobal(28u, 0u, 40u, 25u, 20u, 0x800);
  const auto global_store_b128_program_words =
      MakeGlobal(29u, 0u, 40u, 28u, 20u, 0xa00);
  const std::array<std::uint32_t, 13> global_store_program_words{
      global_store_b8_words[0],
      global_store_b8_words[1],
      global_store_b16_words[0],
      global_store_b16_words[1],
      global_store_b32_program_words[0],
      global_store_b32_program_words[1],
      global_store_b64_words[0],
      global_store_b64_words[1],
      global_store_b96_words[0],
      global_store_b96_words[1],
      global_store_b128_program_words[0],
      global_store_b128_program_words[1],
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> global_store_program;
  if (!Expect(decoder.DecodeProgram(global_store_program_words,
                                    &global_store_program, &error_message),
              "expected GLOBAL store program decode success") ||
      !Expect(global_store_program.size() == 7u,
              "expected seven decoded GLOBAL store instructions") ||
      !Expect(global_store_program[0].opcode == "GLOBAL_STORE_B8",
              "expected decoded GLOBAL_STORE_B8 opcode") ||
      !Expect(global_store_program[1].opcode == "GLOBAL_STORE_B16",
              "expected decoded GLOBAL_STORE_B16 opcode") ||
      !Expect(global_store_program[2].opcode == "GLOBAL_STORE_B32",
              "expected decoded GLOBAL_STORE_B32 opcode") ||
      !Expect(global_store_program[3].opcode == "GLOBAL_STORE_B64",
              "expected decoded GLOBAL_STORE_B64 opcode") ||
      !Expect(global_store_program[4].opcode == "GLOBAL_STORE_B96",
              "expected decoded GLOBAL_STORE_B96 opcode") ||
      !Expect(global_store_program[5].opcode == "GLOBAL_STORE_B128",
              "expected decoded GLOBAL_STORE_B128 opcode") ||
      !Expect(global_store_program[6].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after GLOBAL stores")) {
    return 1;
  }

  auto initialize_global_store_state = [](WaveExecutionState* state) {
    state->exec_mask = 0x80000005ull;
    state->sgprs[20] = 0x4000u;
    for (std::size_t lane = 0; lane < 32u; ++lane) {
      state->vgprs[40][lane] = static_cast<std::uint32_t>(lane * 16u);
      state->vgprs[20][lane] = static_cast<std::uint32_t>(0x40u + lane);
      state->vgprs[21][lane] = static_cast<std::uint32_t>(0x8100u + lane);
      state->vgprs[22][lane] = 0x10000000u + static_cast<std::uint32_t>(lane);
      state->vgprs[23][lane] = 0x20000000u + static_cast<std::uint32_t>(lane);
      state->vgprs[24][lane] = 0x30000000u + static_cast<std::uint32_t>(lane);
      state->vgprs[25][lane] = 0x40000000u + static_cast<std::uint32_t>(lane);
      state->vgprs[26][lane] = 0x50000000u + static_cast<std::uint32_t>(lane);
      state->vgprs[27][lane] = 0x60000000u + static_cast<std::uint32_t>(lane);
      state->vgprs[28][lane] = 0x70000000u + static_cast<std::uint32_t>(lane);
      state->vgprs[29][lane] = 0x71000000u + static_cast<std::uint32_t>(lane);
      state->vgprs[30][lane] = 0x72000000u + static_cast<std::uint32_t>(lane);
      state->vgprs[31][lane] = 0x73000000u + static_cast<std::uint32_t>(lane);
    }
  };
  auto expect_global_store_state = [](const WaveExecutionState& state) {
    if (!(state.lane_count == 32u && state.exec_mask == 0x80000005ull &&
          state.sgprs[20] == 0x4000u && state.halted &&
          !state.waiting_on_barrier && state.pc == 6u)) {
      return false;
    }
    for (std::size_t lane = 0; lane < 32u; ++lane) {
      if (state.vgprs[40][lane] != static_cast<std::uint32_t>(lane * 16u) ||
          state.vgprs[20][lane] != static_cast<std::uint32_t>(0x40u + lane) ||
          state.vgprs[21][lane] != static_cast<std::uint32_t>(0x8100u + lane) ||
          state.vgprs[22][lane] != 0x10000000u + static_cast<std::uint32_t>(lane) ||
          state.vgprs[23][lane] != 0x20000000u + static_cast<std::uint32_t>(lane) ||
          state.vgprs[24][lane] != 0x30000000u + static_cast<std::uint32_t>(lane) ||
          state.vgprs[25][lane] != 0x40000000u + static_cast<std::uint32_t>(lane) ||
          state.vgprs[26][lane] != 0x50000000u + static_cast<std::uint32_t>(lane) ||
          state.vgprs[27][lane] != 0x60000000u + static_cast<std::uint32_t>(lane) ||
          state.vgprs[28][lane] != 0x70000000u + static_cast<std::uint32_t>(lane) ||
          state.vgprs[29][lane] != 0x71000000u + static_cast<std::uint32_t>(lane) ||
          state.vgprs[30][lane] != 0x72000000u + static_cast<std::uint32_t>(lane) ||
          state.vgprs[31][lane] != 0x73000000u + static_cast<std::uint32_t>(lane)) {
        return false;
      }
    }
    return true;
  };
  auto expect_global_store_memory = [](LinearExecutionMemory* memory) {
    constexpr std::uint64_t kActiveMask = 0x80000005ull;
    for (std::uint32_t lane = 0; lane < 32u; ++lane) {
      const bool active = (kActiveMask & (1ull << lane)) != 0u;
      const std::uint64_t lane_address = 0x4000u + lane * 16u;

      std::uint8_t b8 = 0;
      std::uint16_t b16 = 0;
      std::uint32_t b32 = 0;
      std::uint32_t b64_lo = 0;
      std::uint32_t b64_hi = 0;
      std::uint32_t b96_0 = 0;
      std::uint32_t b96_1 = 0;
      std::uint32_t b96_2 = 0;
      std::uint32_t b128_0 = 0;
      std::uint32_t b128_1 = 0;
      std::uint32_t b128_2 = 0;
      std::uint32_t b128_3 = 0;
      if (!memory->LoadU8(lane_address + 0x000u, &b8) ||
          !memory->LoadU16(lane_address + 0x200u, &b16) ||
          !memory->LoadU32(lane_address + 0x400u, &b32) ||
          !memory->LoadU32(lane_address + 0x600u, &b64_lo) ||
          !memory->LoadU32(lane_address + 0x604u, &b64_hi) ||
          !memory->LoadU32(lane_address + 0x800u, &b96_0) ||
          !memory->LoadU32(lane_address + 0x804u, &b96_1) ||
          !memory->LoadU32(lane_address + 0x808u, &b96_2) ||
          !memory->LoadU32(lane_address + 0xa00u, &b128_0) ||
          !memory->LoadU32(lane_address + 0xa04u, &b128_1) ||
          !memory->LoadU32(lane_address + 0xa08u, &b128_2) ||
          !memory->LoadU32(lane_address + 0xa0cu, &b128_3)) {
        return false;
      }

      if (!active) {
        if (b8 != 0u || b16 != 0u || b32 != 0u || b64_lo != 0u || b64_hi != 0u ||
            b96_0 != 0u || b96_1 != 0u || b96_2 != 0u || b128_0 != 0u ||
            b128_1 != 0u || b128_2 != 0u || b128_3 != 0u) {
          return false;
        }
        continue;
      }

      if (b8 != static_cast<std::uint8_t>(0x40u + lane) ||
          b16 != static_cast<std::uint16_t>(0x8100u + lane) ||
          b32 != 0x10000000u + lane || b64_lo != 0x20000000u + lane ||
          b64_hi != 0x30000000u + lane || b96_0 != 0x40000000u + lane ||
          b96_1 != 0x50000000u + lane || b96_2 != 0x60000000u + lane ||
          b128_0 != 0x70000000u + lane || b128_1 != 0x71000000u + lane ||
          b128_2 != 0x72000000u + lane || b128_3 != 0x73000000u + lane) {
        return false;
      }
    }
    return true;
  };

  LinearExecutionMemory decoded_global_store_memory(0x1000u, 0x4000u);
  WaveExecutionState decoded_global_store_state;
  initialize_global_store_state(&decoded_global_store_state);
  if (!Expect(interpreter.ExecuteProgram(global_store_program,
                                         &decoded_global_store_state,
                                         &decoded_global_store_memory,
                                         &error_message),
              "expected decoded GLOBAL store execution success") ||
      !Expect(expect_global_store_state(decoded_global_store_state),
              "expected decoded GLOBAL store register state") ||
      !Expect(expect_global_store_memory(&decoded_global_store_memory),
              "expected decoded GLOBAL store memory state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_global_store_program;
  if (!Expect(interpreter.CompileProgram(global_store_program,
                                         &compiled_global_store_program,
                                         &error_message),
              "expected compiled GLOBAL store program success") ||
      !Expect(compiled_global_store_program.size() == 7u,
              "expected seven compiled GLOBAL store instructions") ||
      !Expect(compiled_global_store_program[0].opcode ==
                  Gfx1201CompiledOpcode::kGlobalStoreB8,
              "expected compiled GLOBAL_STORE_B8 opcode") ||
      !Expect(compiled_global_store_program[1].opcode ==
                  Gfx1201CompiledOpcode::kGlobalStoreB16,
              "expected compiled GLOBAL_STORE_B16 opcode") ||
      !Expect(compiled_global_store_program[2].opcode ==
                  Gfx1201CompiledOpcode::kGlobalStoreB32,
              "expected compiled GLOBAL_STORE_B32 opcode") ||
      !Expect(compiled_global_store_program[3].opcode ==
                  Gfx1201CompiledOpcode::kGlobalStoreB64,
              "expected compiled GLOBAL_STORE_B64 opcode") ||
      !Expect(compiled_global_store_program[4].opcode ==
                  Gfx1201CompiledOpcode::kGlobalStoreB96,
              "expected compiled GLOBAL_STORE_B96 opcode") ||
      !Expect(compiled_global_store_program[5].opcode ==
                  Gfx1201CompiledOpcode::kGlobalStoreB128,
              "expected compiled GLOBAL_STORE_B128 opcode") ||
      !Expect(compiled_global_store_program[6].opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after GLOBAL stores")) {
    return 1;
  }

  LinearExecutionMemory compiled_global_store_memory(0x1000u, 0x4000u);
  WaveExecutionState compiled_global_store_state;
  initialize_global_store_state(&compiled_global_store_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_global_store_program,
                                         &compiled_global_store_state,
                                         &compiled_global_store_memory,
                                         &error_message),
              "expected compiled GLOBAL store execution success") ||
      !Expect(expect_global_store_state(compiled_global_store_state),
              "expected compiled GLOBAL store register state") ||
      !Expect(expect_global_store_memory(&compiled_global_store_memory),
              "expected compiled GLOBAL store memory state")) {
    return 1;
  }

  const auto global_store_d16_hi_b8_program_words =
      MakeGlobal(36u, 0u, 40u, 20u, 20u, 0x000);
  const auto global_store_d16_hi_b16_program_words =
      MakeGlobal(37u, 0u, 40u, 21u, 20u, 0x200);
  const std::array<std::uint32_t, 5> global_store_d16_program_words{
      global_store_d16_hi_b8_program_words[0],
      global_store_d16_hi_b8_program_words[1],
      global_store_d16_hi_b16_program_words[0],
      global_store_d16_hi_b16_program_words[1],
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> global_store_d16_program;
  if (!Expect(decoder.DecodeProgram(global_store_d16_program_words,
                                    &global_store_d16_program, &error_message),
              "expected GLOBAL D16 store program decode success") ||
      !Expect(global_store_d16_program.size() == 3u,
              "expected three decoded GLOBAL D16 store instructions") ||
      !Expect(global_store_d16_program[0].opcode == "GLOBAL_STORE_D16_HI_B8",
              "expected decoded GLOBAL_STORE_D16_HI_B8 opcode") ||
      !Expect(global_store_d16_program[1].opcode == "GLOBAL_STORE_D16_HI_B16",
              "expected decoded GLOBAL_STORE_D16_HI_B16 opcode") ||
      !Expect(global_store_d16_program[2].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after GLOBAL D16 stores")) {
    return 1;
  }

  auto initialize_global_store_d16_state = [](WaveExecutionState* state) {
    state->exec_mask = 0x80000005ull;
    state->sgprs[20] = 0x6000u;
    for (std::size_t lane = 0; lane < 32u; ++lane) {
      state->vgprs[40][lane] = static_cast<std::uint32_t>(lane * 16u);
      state->vgprs[20][lane] =
          0xab000000u |
          (static_cast<std::uint32_t>(0x40u + lane) << 16) | 0x1234u;
      state->vgprs[21][lane] =
          (static_cast<std::uint32_t>(0x5000u + lane) << 16) | 0x5678u;
    }
  };
  auto expect_global_store_d16_state = [](const WaveExecutionState& state) {
    if (!(state.lane_count == 32u && state.exec_mask == 0x80000005ull &&
          state.sgprs[20] == 0x6000u && state.halted &&
          !state.waiting_on_barrier && state.pc == 2u)) {
      return false;
    }
    for (std::size_t lane = 0; lane < 32u; ++lane) {
      const std::uint32_t expected_b8 =
          0xab000000u |
          (static_cast<std::uint32_t>(0x40u + lane) << 16) | 0x1234u;
      const std::uint32_t expected_b16 =
          (static_cast<std::uint32_t>(0x5000u + lane) << 16) | 0x5678u;
      if (state.vgprs[40][lane] != static_cast<std::uint32_t>(lane * 16u) ||
          state.vgprs[20][lane] != expected_b8 ||
          state.vgprs[21][lane] != expected_b16) {
        return false;
      }
    }
    return true;
  };
  auto expect_global_store_d16_memory = [](LinearExecutionMemory* memory) {
    constexpr std::uint64_t kActiveMask = 0x80000005ull;
    for (std::uint32_t lane = 0; lane < 32u; ++lane) {
      const bool active = (kActiveMask & (1ull << lane)) != 0u;
      const std::uint64_t lane_address = 0x6000u + lane * 16u;
      std::uint8_t b8 = 0;
      std::uint16_t b16 = 0;
      if (!memory->LoadU8(lane_address + 0x000u, &b8) ||
          !memory->LoadU16(lane_address + 0x200u, &b16)) {
        return false;
      }
      if (!active) {
        if (b8 != 0u || b16 != 0u) {
          return false;
        }
        continue;
      }
      if (b8 != static_cast<std::uint8_t>(0x40u + lane) ||
          b16 != static_cast<std::uint16_t>(0x5000u + lane)) {
        return false;
      }
    }
    return true;
  };

  LinearExecutionMemory decoded_global_store_d16_memory(0x1000u, 0x6000u);
  WaveExecutionState decoded_global_store_d16_state;
  initialize_global_store_d16_state(&decoded_global_store_d16_state);
  if (!Expect(interpreter.ExecuteProgram(global_store_d16_program,
                                         &decoded_global_store_d16_state,
                                         &decoded_global_store_d16_memory,
                                         &error_message),
              "expected decoded GLOBAL D16 store execution success") ||
      !Expect(expect_global_store_d16_state(decoded_global_store_d16_state),
              "expected decoded GLOBAL D16 store state") ||
      !Expect(expect_global_store_d16_memory(&decoded_global_store_d16_memory),
              "expected decoded GLOBAL D16 store memory state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_global_store_d16_program;
  if (!Expect(interpreter.CompileProgram(global_store_d16_program,
                                         &compiled_global_store_d16_program,
                                         &error_message),
              "expected compiled GLOBAL D16 store program success") ||
      !Expect(compiled_global_store_d16_program.size() == 3u,
              "expected three compiled GLOBAL D16 store instructions") ||
      !Expect(compiled_global_store_d16_program[0].opcode ==
                  Gfx1201CompiledOpcode::kGlobalStoreD16HiB8,
              "expected compiled GLOBAL_STORE_D16_HI_B8 opcode") ||
      !Expect(compiled_global_store_d16_program[1].opcode ==
                  Gfx1201CompiledOpcode::kGlobalStoreD16HiB16,
              "expected compiled GLOBAL_STORE_D16_HI_B16 opcode") ||
      !Expect(compiled_global_store_d16_program[2].opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after GLOBAL D16 stores")) {
    return 1;
  }

  LinearExecutionMemory compiled_global_store_d16_memory(0x1000u, 0x6000u);
  WaveExecutionState compiled_global_store_d16_state;
  initialize_global_store_d16_state(&compiled_global_store_d16_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_global_store_d16_program,
                                         &compiled_global_store_d16_state,
                                         &compiled_global_store_d16_memory,
                                         &error_message),
              "expected compiled GLOBAL D16 store execution success") ||
      !Expect(expect_global_store_d16_state(compiled_global_store_d16_state),
              "expected compiled GLOBAL D16 store state") ||
      !Expect(expect_global_store_d16_memory(&compiled_global_store_d16_memory),
              "expected compiled GLOBAL D16 store memory state")) {
    return 1;
  }

  if (!RunGlobalAtomicU32BatchTest(decoder, interpreter, &error_message)) {
    return 1;
  }
  if (!RunGlobalAtomicU64BatchTest(decoder, interpreter, &error_message)) {
    return 1;
  }
  if (!RunGlobalAtomicF32BatchTest(decoder, interpreter, &error_message)) {
    return 1;
  }
  if (!RunGlobalAtomicPackedBatchTest(decoder, interpreter, &error_message)) {
    return 1;
  }
  if (!RunDsBatchTest(decoder, interpreter, &error_message)) {
    return 1;
  }

  const auto load_b32_words = MakeSmem(0u, 40u, 2u, true, 4u);
  const auto load_b64_words = MakeSmem(1u, 42u, 4u, false, 21u, true);
  const auto load_b96_words = MakeSmem(5u, 48u, 8u, true, 12u);
  const auto load_b128_words = MakeSmem(2u, 52u, 10u, false, 22u, true);
  const auto load_b256_words = MakeSmem(3u, 56u, 12u, true, 20u);
  const auto load_b512_words = MakeSmem(4u, 64u, 14u, false, 23u, true);
  const auto load_i8_words = MakeSmem(8u, 44u, 6u, true, 0u);
  const auto load_u8_words = MakeSmem(9u, 45u, 6u, true, 1u);
  const auto load_i16_words = MakeSmem(10u, 46u, 6u, true, 2u);
  const auto load_u16_words = MakeSmem(11u, 47u, 6u, true, 4u);

  DecodedInstruction load_b32_instruction;
  std::size_t load_words_consumed = 0;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(load_b32_words.data(),
                                                 load_b32_words.size()),
                  &load_b32_instruction, &load_words_consumed, &error_message),
              "expected S_LOAD_B32 direct decode success") ||
      !Expect(ExpectThreeOperandInstruction(load_b32_instruction, "S_LOAD_B32",
                                            OperandKind::kSgpr, 40u,
                                            OperandKind::kSgpr, 2u,
                                            OperandKind::kImm32, 4u),
              "expected decoded S_LOAD_B32 operands") ||
      !Expect(load_words_consumed == 2u,
              "expected S_LOAD_B32 to consume two dwords") ||
      !Expect(ExpectOperandDescriptor(
                  load_b32_instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kScalarDestination,
                  OperandValueClass::kScalarRegister, OperandAccess::kWrite,
                  FragmentKind::kScalar, 32u, 1u, false),
              "expected S_LOAD_B32 destination descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  load_b32_instruction.operands[1], OperandRole::kSource0,
                  OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 64u, 2u, false),
              "expected S_LOAD_B32 base descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  load_b32_instruction.operands[2], OperandRole::kSource1,
                  OperandSlotKind::kSource1, OperandValueClass::kUnknown,
                  OperandAccess::kRead, FragmentKind::kScalar, 32u, 1u, false),
              "expected S_LOAD_B32 offset descriptor")) {
    return 1;
  }

  DecodedInstruction load_b64_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(load_b64_words.data(),
                                                 load_b64_words.size()),
                  &load_b64_instruction, &load_words_consumed, &error_message),
              "expected S_LOAD_B64 direct decode success") ||
      !Expect(ExpectThreeOperandInstruction(load_b64_instruction, "S_LOAD_B64",
                                            OperandKind::kSgpr, 42u,
                                            OperandKind::kSgpr, 4u,
                                            OperandKind::kSgpr, 21u),
              "expected decoded S_LOAD_B64 operands") ||
      !Expect(load_words_consumed == 2u,
              "expected S_LOAD_B64 to consume two dwords") ||
      !Expect(ExpectOperandDescriptor(
                  load_b64_instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kScalarDestination,
                  OperandValueClass::kScalarRegister, OperandAccess::kWrite,
                  FragmentKind::kScalar, 64u, 2u, false),
              "expected S_LOAD_B64 destination descriptor")) {
    return 1;
  }

  DecodedInstruction load_b96_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(load_b96_words.data(),
                                                 load_b96_words.size()),
                  &load_b96_instruction, &load_words_consumed, &error_message),
              "expected S_LOAD_B96 direct decode success") ||
      !Expect(ExpectThreeOperandInstruction(load_b96_instruction, "S_LOAD_B96",
                                            OperandKind::kSgpr, 48u,
                                            OperandKind::kSgpr, 8u,
                                            OperandKind::kImm32, 12u),
              "expected decoded S_LOAD_B96 operands") ||
      !Expect(load_words_consumed == 2u,
              "expected S_LOAD_B96 to consume two dwords") ||
      !Expect(ExpectOperandDescriptor(
                  load_b96_instruction.operands[0], OperandRole::kDestination,
                  OperandSlotKind::kScalarDestination,
                  OperandValueClass::kScalarRegister, OperandAccess::kWrite,
                  FragmentKind::kScalar, 32u, 3u, false),
              "expected S_LOAD_B96 destination descriptor")) {
    return 1;
  }

  DecodedInstruction load_b128_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(load_b128_words.data(),
                                                 load_b128_words.size()),
                  &load_b128_instruction, &load_words_consumed,
                  &error_message),
              "expected S_LOAD_B128 direct decode success") ||
      !Expect(ExpectThreeOperandInstruction(load_b128_instruction,
                                            "S_LOAD_B128", OperandKind::kSgpr,
                                            52u, OperandKind::kSgpr, 10u,
                                            OperandKind::kSgpr, 22u),
              "expected decoded S_LOAD_B128 operands") ||
      !Expect(load_words_consumed == 2u,
              "expected S_LOAD_B128 to consume two dwords") ||
      !Expect(ExpectOperandDescriptor(
                  load_b128_instruction.operands[0],
                  OperandRole::kDestination,
                  OperandSlotKind::kScalarDestination,
                  OperandValueClass::kScalarRegister, OperandAccess::kWrite,
                  FragmentKind::kScalar, 128u, 4u, false),
              "expected S_LOAD_B128 destination descriptor")) {
    return 1;
  }

  DecodedInstruction load_i8_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(load_i8_words.data(),
                                                 load_i8_words.size()),
                  &load_i8_instruction, &load_words_consumed, &error_message),
              "expected S_LOAD_I8 direct decode success") ||
      !Expect(ExpectThreeOperandInstruction(load_i8_instruction, "S_LOAD_I8",
                                            OperandKind::kSgpr, 44u,
                                            OperandKind::kSgpr, 6u,
                                            OperandKind::kImm32, 0u),
              "expected decoded S_LOAD_I8 operands") ||
      !Expect(load_words_consumed == 2u,
              "expected S_LOAD_I8 to consume two dwords")) {
    return 1;
  }

  const std::array<std::uint32_t, 21> smem_load_program_words{{
      load_b32_words[0],  load_b32_words[1],  load_b64_words[0],
      load_b64_words[1],  load_b96_words[0],  load_b96_words[1],
      load_b128_words[0], load_b128_words[1], load_b256_words[0],
      load_b256_words[1], load_b512_words[0], load_b512_words[1],
      load_i8_words[0],   load_i8_words[1],   load_u8_words[0],
      load_u8_words[1],   load_i16_words[0],  load_i16_words[1],
      load_u16_words[0],  load_u16_words[1],  MakeSopp(48u),
  }};
  std::vector<DecodedInstruction> smem_load_program;
  if (!Expect(decoder.DecodeProgram(smem_load_program_words, &smem_load_program,
                                    &error_message),
              "expected SMEM load program decode success") ||
      !Expect(smem_load_program.size() == 11u,
              "expected eleven decoded SMEM load instructions") ||
      !Expect(smem_load_program[0].opcode == "S_LOAD_B32",
              "expected decoded S_LOAD_B32 program opcode") ||
      !Expect(smem_load_program[1].opcode == "S_LOAD_B64",
              "expected decoded S_LOAD_B64 program opcode") ||
      !Expect(smem_load_program[2].opcode == "S_LOAD_B96",
              "expected decoded S_LOAD_B96 program opcode") ||
      !Expect(smem_load_program[3].opcode == "S_LOAD_B128",
              "expected decoded S_LOAD_B128 program opcode") ||
      !Expect(smem_load_program[4].opcode == "S_LOAD_B256",
              "expected decoded S_LOAD_B256 program opcode") ||
      !Expect(smem_load_program[5].opcode == "S_LOAD_B512",
              "expected decoded S_LOAD_B512 program opcode") ||
      !Expect(smem_load_program[6].opcode == "S_LOAD_I8",
              "expected decoded S_LOAD_I8 program opcode") ||
      !Expect(smem_load_program[7].opcode == "S_LOAD_U8",
              "expected decoded S_LOAD_U8 program opcode") ||
      !Expect(smem_load_program[8].opcode == "S_LOAD_I16",
              "expected decoded S_LOAD_I16 program opcode") ||
      !Expect(smem_load_program[9].opcode == "S_LOAD_U16",
              "expected decoded S_LOAD_U16 program opcode") ||
      !Expect(smem_load_program[10].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after SMEM loads")) {
    return 1;
  }

  const std::array<std::uint32_t, 3> expected_load_b96{{
      0x11111111u,
      0x22222222u,
      0x33333333u,
  }};
  const std::array<std::uint32_t, 4> expected_load_b128{{
      0x44444444u,
      0x55555555u,
      0x66666666u,
      0x77777777u,
  }};
  const std::array<std::uint32_t, 8> expected_load_b256{{
      0x88000001u,
      0x88000002u,
      0x88000003u,
      0x88000004u,
      0x88000005u,
      0x88000006u,
      0x88000007u,
      0x88000008u,
  }};
  const std::array<std::uint32_t, 16> expected_load_b512{{
      0x99000001u,
      0x99000002u,
      0x99000003u,
      0x99000004u,
      0x99000005u,
      0x99000006u,
      0x99000007u,
      0x99000008u,
      0x99000009u,
      0x9900000au,
      0x9900000bu,
      0x9900000cu,
      0x9900000du,
      0x9900000eu,
      0x9900000fu,
      0x99000010u,
  }};

  auto initialize_smem_load_state = [](WaveExecutionState* state) {
    state->exec_mask = 0x5u;
    state->sgprs[2] = 0x1000u;
    state->sgprs[3] = 0u;
    state->sgprs[4] = 0x1010u;
    state->sgprs[5] = 0u;
    state->sgprs[6] = 0x1020u;
    state->sgprs[7] = 0u;
    state->sgprs[8] = 0x1040u;
    state->sgprs[9] = 0u;
    state->sgprs[10] = 0x1060u;
    state->sgprs[11] = 0u;
    state->sgprs[12] = 0x1080u;
    state->sgprs[13] = 0u;
    state->sgprs[14] = 0x1100u;
    state->sgprs[15] = 0u;
    state->sgprs[21] = 8u;
    state->sgprs[22] = 16u;
    state->sgprs[23] = 12u;
  };
  auto expect_smem_load_state = [&](const WaveExecutionState& state) {
    if (!(state.lane_count == 32u && state.exec_mask == 0x5u &&
          state.sgprs[40] == 0x11223344u &&
          state.sgprs[42] == 0x55667788u &&
          state.sgprs[43] == 0x99aabbccu &&
          state.sgprs[44] == 0xffffff80u &&
          state.sgprs[45] == 0x7fu &&
          state.sgprs[46] == 0xffff8001u &&
          state.sgprs[47] == 0x8123u && state.halted &&
          !state.waiting_on_barrier && state.pc == 10u)) {
      return false;
    }
    for (std::size_t i = 0; i < expected_load_b96.size(); ++i) {
      if (state.sgprs[48u + i] != expected_load_b96[i]) {
        return false;
      }
    }
    for (std::size_t i = 0; i < expected_load_b128.size(); ++i) {
      if (state.sgprs[52u + i] != expected_load_b128[i]) {
        return false;
      }
    }
    for (std::size_t i = 0; i < expected_load_b256.size(); ++i) {
      if (state.sgprs[56u + i] != expected_load_b256[i]) {
        return false;
      }
    }
    for (std::size_t i = 0; i < expected_load_b512.size(); ++i) {
      if (state.sgprs[64u + i] != expected_load_b512[i]) {
        return false;
      }
    }
    return true;
  };
  LinearExecutionMemory smem_load_memory(0x200u, 0x1000u);
  if (!Expect(smem_load_memory.WriteU32(0x1004u, 0x11223344u),
              "expected SMEM test write for S_LOAD_B32") ||
      !Expect(smem_load_memory.WriteU32(0x1018u, 0x55667788u),
              "expected SMEM test write for S_LOAD_B64 low") ||
      !Expect(smem_load_memory.WriteU32(0x101cu, 0x99aabbccu),
              "expected SMEM test write for S_LOAD_B64 high") ||
      !Expect(smem_load_memory.StoreU8(0x1020u, 0x80u),
              "expected SMEM test write for S_LOAD_I8") ||
      !Expect(smem_load_memory.StoreU8(0x1021u, 0x7fu),
              "expected SMEM test write for S_LOAD_U8") ||
      !Expect(smem_load_memory.StoreU16(0x1022u, 0x8001u),
              "expected SMEM test write for S_LOAD_I16") ||
      !Expect(smem_load_memory.StoreU16(0x1024u, 0x8123u),
              "expected SMEM test write for S_LOAD_U16")) {
    return 1;
  }
  for (std::size_t i = 0; i < expected_load_b96.size(); ++i) {
    if (!Expect(smem_load_memory.WriteU32(
                    0x104cu + static_cast<std::uint32_t>(i * 4u),
                    expected_load_b96[i]),
                "expected SMEM test write for S_LOAD_B96")) {
      return 1;
    }
  }
  for (std::size_t i = 0; i < expected_load_b128.size(); ++i) {
    if (!Expect(smem_load_memory.WriteU32(
                    0x1070u + static_cast<std::uint32_t>(i * 4u),
                    expected_load_b128[i]),
                "expected SMEM test write for S_LOAD_B128")) {
      return 1;
    }
  }
  for (std::size_t i = 0; i < expected_load_b256.size(); ++i) {
    if (!Expect(smem_load_memory.WriteU32(
                    0x1094u + static_cast<std::uint32_t>(i * 4u),
                    expected_load_b256[i]),
                "expected SMEM test write for S_LOAD_B256")) {
      return 1;
    }
  }
  for (std::size_t i = 0; i < expected_load_b512.size(); ++i) {
    if (!Expect(smem_load_memory.WriteU32(
                    0x110cu + static_cast<std::uint32_t>(i * 4u),
                    expected_load_b512[i]),
                "expected SMEM test write for S_LOAD_B512")) {
      return 1;
    }
  }

  WaveExecutionState decoded_smem_load_state;
  initialize_smem_load_state(&decoded_smem_load_state);
  if (!Expect(interpreter.ExecuteProgram(smem_load_program,
                                         &decoded_smem_load_state,
                                         &smem_load_memory, &error_message),
              "expected decoded SMEM load execution success") ||
      !Expect(expect_smem_load_state(decoded_smem_load_state),
              "expected decoded SMEM load state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_smem_load_program;
  if (!Expect(interpreter.CompileProgram(smem_load_program,
                                         &compiled_smem_load_program,
                                         &error_message),
              "expected compiled SMEM load program success") ||
      !Expect(compiled_smem_load_program.size() == 11u,
              "expected eleven compiled SMEM load instructions") ||
      !Expect(compiled_smem_load_program[0].opcode ==
                  Gfx1201CompiledOpcode::kSLoadB32,
              "expected compiled S_LOAD_B32 opcode") ||
      !Expect(compiled_smem_load_program[1].opcode ==
                  Gfx1201CompiledOpcode::kSLoadB64,
              "expected compiled S_LOAD_B64 opcode") ||
      !Expect(compiled_smem_load_program[2].opcode ==
                  Gfx1201CompiledOpcode::kSLoadB96,
              "expected compiled S_LOAD_B96 opcode") ||
      !Expect(compiled_smem_load_program[3].opcode ==
                  Gfx1201CompiledOpcode::kSLoadB128,
              "expected compiled S_LOAD_B128 opcode") ||
      !Expect(compiled_smem_load_program[4].opcode ==
                  Gfx1201CompiledOpcode::kSLoadB256,
              "expected compiled S_LOAD_B256 opcode") ||
      !Expect(compiled_smem_load_program[5].opcode ==
                  Gfx1201CompiledOpcode::kSLoadB512,
              "expected compiled S_LOAD_B512 opcode") ||
      !Expect(compiled_smem_load_program[6].opcode ==
                  Gfx1201CompiledOpcode::kSLoadI8,
              "expected compiled S_LOAD_I8 opcode") ||
      !Expect(compiled_smem_load_program[7].opcode ==
                  Gfx1201CompiledOpcode::kSLoadU8,
              "expected compiled S_LOAD_U8 opcode") ||
      !Expect(compiled_smem_load_program[8].opcode ==
                  Gfx1201CompiledOpcode::kSLoadI16,
              "expected compiled S_LOAD_I16 opcode") ||
      !Expect(compiled_smem_load_program[9].opcode ==
                  Gfx1201CompiledOpcode::kSLoadU16,
              "expected compiled S_LOAD_U16 opcode") ||
      !Expect(compiled_smem_load_program[10].opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after SMEM loads")) {
    return 1;
  }

  WaveExecutionState compiled_smem_load_state;
  initialize_smem_load_state(&compiled_smem_load_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_smem_load_program,
                                         &compiled_smem_load_state,
                                         &smem_load_memory, &error_message),
              "expected compiled SMEM load execution success") ||
      !Expect(expect_smem_load_state(compiled_smem_load_state),
              "expected compiled SMEM load state")) {
    return 1;
  }

  const auto buffer_load_b32_words = MakeSmemBufferLoad(16u, 76u, 40u, 4, 30u);
  const auto buffer_load_b64_words = MakeSmemBufferLoad(17u, 77u, 44u, 8, 31u);
  const auto buffer_load_b96_words = MakeSmemBufferLoad(21u, 79u, 48u, 16, 32u);
  const auto buffer_load_b128_words =
      MakeSmemBufferLoad(18u, 82u, 52u, 12, 33u);
  const auto buffer_load_b256_words =
      MakeSmemBufferLoad(19u, 86u, 56u, 24, 34u);
  const auto buffer_load_b512_words =
      MakeSmemBufferLoad(20u, 94u, 60u, 28, 35u);
  const auto buffer_load_i8_words = MakeSmemBufferLoad(24u, 72u, 64u, 0, 36u);
  const auto buffer_load_u8_words = MakeSmemBufferLoad(25u, 73u, 64u, 1, 36u);
  const auto buffer_load_i16_words =
      MakeSmemBufferLoad(26u, 74u, 64u, 2, 36u);
  const auto buffer_load_u16_words =
      MakeSmemBufferLoad(27u, 75u, 64u, 4, 36u);

  DecodedInstruction buffer_load_b32_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(buffer_load_b32_words.data(),
                                                 buffer_load_b32_words.size()),
                  &buffer_load_b32_instruction, &load_words_consumed,
                  &error_message),
              "expected S_BUFFER_LOAD_B32 direct decode success") ||
      !Expect(ExpectFourOperandInstruction(
                  buffer_load_b32_instruction, "S_BUFFER_LOAD_B32",
                  OperandKind::kSgpr, 76u, OperandKind::kSgpr, 40u,
                  OperandKind::kImm32, 4u, OperandKind::kSgpr, 30u),
              "expected decoded S_BUFFER_LOAD_B32 operands") ||
      !Expect(load_words_consumed == 2u,
              "expected S_BUFFER_LOAD_B32 to consume two dwords") ||
      !Expect(ExpectOperandDescriptor(
                  buffer_load_b32_instruction.operands[1],
                  OperandRole::kSource0, OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 128u, 4u, false),
              "expected S_BUFFER_LOAD_B32 base descriptor")) {
    return 1;
  }

  DecodedInstruction buffer_load_b128_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(buffer_load_b128_words.data(),
                                                 buffer_load_b128_words.size()),
                  &buffer_load_b128_instruction, &load_words_consumed,
                  &error_message),
              "expected S_BUFFER_LOAD_B128 direct decode success") ||
      !Expect(ExpectFourOperandInstruction(
                  buffer_load_b128_instruction, "S_BUFFER_LOAD_B128",
                  OperandKind::kSgpr, 82u, OperandKind::kSgpr, 52u,
                  OperandKind::kImm32, 12u, OperandKind::kSgpr, 33u),
              "expected decoded S_BUFFER_LOAD_B128 operands") ||
      !Expect(load_words_consumed == 2u,
              "expected S_BUFFER_LOAD_B128 to consume two dwords") ||
      !Expect(ExpectOperandDescriptor(
                  buffer_load_b128_instruction.operands[0],
                  OperandRole::kDestination,
                  OperandSlotKind::kScalarDestination,
                  OperandValueClass::kScalarRegister, OperandAccess::kWrite,
                  FragmentKind::kScalar, 128u, 4u, false),
              "expected S_BUFFER_LOAD_B128 destination descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 21> buffer_load_program_words{{
      buffer_load_b32_words[0],  buffer_load_b32_words[1],
      buffer_load_b64_words[0],  buffer_load_b64_words[1],
      buffer_load_b96_words[0],  buffer_load_b96_words[1],
      buffer_load_b128_words[0], buffer_load_b128_words[1],
      buffer_load_b256_words[0], buffer_load_b256_words[1],
      buffer_load_b512_words[0], buffer_load_b512_words[1],
      buffer_load_i8_words[0],   buffer_load_i8_words[1],
      buffer_load_u8_words[0],   buffer_load_u8_words[1],
      buffer_load_i16_words[0],  buffer_load_i16_words[1],
      buffer_load_u16_words[0],  buffer_load_u16_words[1],
      MakeSopp(48u),
  }};
  std::vector<DecodedInstruction> buffer_load_program;
  if (!Expect(decoder.DecodeProgram(buffer_load_program_words,
                                    &buffer_load_program, &error_message),
              "expected buffer SMEM load program decode success") ||
      !Expect(buffer_load_program.size() == 11u,
              "expected eleven decoded buffer SMEM load instructions") ||
      !Expect(buffer_load_program[0].opcode == "S_BUFFER_LOAD_B32",
              "expected decoded S_BUFFER_LOAD_B32 program opcode") ||
      !Expect(buffer_load_program[1].opcode == "S_BUFFER_LOAD_B64",
              "expected decoded S_BUFFER_LOAD_B64 program opcode") ||
      !Expect(buffer_load_program[2].opcode == "S_BUFFER_LOAD_B96",
              "expected decoded S_BUFFER_LOAD_B96 program opcode") ||
      !Expect(buffer_load_program[3].opcode == "S_BUFFER_LOAD_B128",
              "expected decoded S_BUFFER_LOAD_B128 program opcode") ||
      !Expect(buffer_load_program[4].opcode == "S_BUFFER_LOAD_B256",
              "expected decoded S_BUFFER_LOAD_B256 program opcode") ||
      !Expect(buffer_load_program[5].opcode == "S_BUFFER_LOAD_B512",
              "expected decoded S_BUFFER_LOAD_B512 program opcode") ||
      !Expect(buffer_load_program[6].opcode == "S_BUFFER_LOAD_I8",
              "expected decoded S_BUFFER_LOAD_I8 program opcode") ||
      !Expect(buffer_load_program[7].opcode == "S_BUFFER_LOAD_U8",
              "expected decoded S_BUFFER_LOAD_U8 program opcode") ||
      !Expect(buffer_load_program[8].opcode == "S_BUFFER_LOAD_I16",
              "expected decoded S_BUFFER_LOAD_I16 program opcode") ||
      !Expect(buffer_load_program[9].opcode == "S_BUFFER_LOAD_U16",
              "expected decoded S_BUFFER_LOAD_U16 program opcode") ||
      !Expect(buffer_load_program[10].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after buffer SMEM loads")) {
    return 1;
  }

  const std::array<std::uint32_t, 3> expected_buffer_load_b96{{
      0x11110001u,
      0x11110002u,
      0x11110003u,
  }};
  const std::array<std::uint32_t, 4> expected_buffer_load_b128{{
      0x22220001u,
      0x22220002u,
      0x22220003u,
      0x22220004u,
  }};
  const std::array<std::uint32_t, 8> expected_buffer_load_b256{{
      0x33330001u,
      0x33330002u,
      0x33330003u,
      0x33330004u,
      0x33330005u,
      0x33330006u,
      0x33330007u,
      0x33330008u,
  }};
  const std::array<std::uint32_t, 16> expected_buffer_load_b512{{
      0x44440001u,
      0x44440002u,
      0x44440003u,
      0x44440004u,
      0x44440005u,
      0x44440006u,
      0x44440007u,
      0x44440008u,
      0x44440009u,
      0x4444000au,
      0x4444000bu,
      0x4444000cu,
      0x4444000du,
      0x4444000eu,
      0x4444000fu,
      0x44440010u,
  }};

  auto initialize_buffer_load_state = [](WaveExecutionState* state) {
    state->exec_mask = 0x3u;
    state->sgprs[40] = 0x2000u;
    state->sgprs[41] = 0u;
    state->sgprs[42] = 0u;
    state->sgprs[43] = 0u;
    state->sgprs[44] = 0x2020u;
    state->sgprs[45] = 0u;
    state->sgprs[46] = 0u;
    state->sgprs[47] = 0u;
    state->sgprs[48] = 0x2040u;
    state->sgprs[49] = 0u;
    state->sgprs[50] = 0u;
    state->sgprs[51] = 0u;
    state->sgprs[52] = 0x2060u;
    state->sgprs[53] = 0u;
    state->sgprs[54] = 0u;
    state->sgprs[55] = 0u;
    state->sgprs[56] = 0x2080u;
    state->sgprs[57] = 0u;
    state->sgprs[58] = 0u;
    state->sgprs[59] = 0u;
    state->sgprs[60] = 0x2100u;
    state->sgprs[61] = 0u;
    state->sgprs[62] = 0u;
    state->sgprs[63] = 0u;
    state->sgprs[64] = 0x2200u;
    state->sgprs[65] = 0u;
    state->sgprs[66] = 0u;
    state->sgprs[67] = 0u;
    state->sgprs[30] = 8u;
    state->sgprs[31] = 12u;
    state->sgprs[32] = 0u;
    state->sgprs[33] = 12u;
    state->sgprs[34] = 16u;
    state->sgprs[35] = 8u;
    state->sgprs[36] = 0u;
  };
  auto expect_buffer_load_state = [&](const WaveExecutionState& state) {
    if (!(state.lane_count == 32u && state.exec_mask == 0x3u &&
          state.sgprs[76] == 0x01234567u &&
          state.sgprs[77] == 0x89abcdefu &&
          state.sgprs[78] == 0x13579bdfu &&
          state.sgprs[72] == 0xffffff81u &&
          state.sgprs[73] == 0x7eu &&
          state.sgprs[74] == 0xffff8002u &&
          state.sgprs[75] == 0x8124u && state.halted &&
          !state.waiting_on_barrier && state.pc == 10u)) {
      return false;
    }
    for (std::size_t i = 0; i < expected_buffer_load_b96.size(); ++i) {
      if (state.sgprs[79u + i] != expected_buffer_load_b96[i]) {
        return false;
      }
    }
    for (std::size_t i = 0; i < expected_buffer_load_b128.size(); ++i) {
      if (state.sgprs[82u + i] != expected_buffer_load_b128[i]) {
        return false;
      }
    }
    for (std::size_t i = 0; i < expected_buffer_load_b256.size(); ++i) {
      if (state.sgprs[86u + i] != expected_buffer_load_b256[i]) {
        return false;
      }
    }
    for (std::size_t i = 0; i < expected_buffer_load_b512.size(); ++i) {
      if (state.sgprs[94u + i] != expected_buffer_load_b512[i]) {
        return false;
      }
    }
    return true;
  };
  LinearExecutionMemory buffer_load_memory(0x300u, 0x2000u);
  if (!Expect(buffer_load_memory.WriteU32(0x200cu, 0x01234567u),
              "expected buffer SMEM test write for S_BUFFER_LOAD_B32") ||
      !Expect(buffer_load_memory.WriteU32(0x2034u, 0x89abcdefu),
              "expected buffer SMEM test write for S_BUFFER_LOAD_B64 low") ||
      !Expect(buffer_load_memory.WriteU32(0x2038u, 0x13579bdfu),
              "expected buffer SMEM test write for S_BUFFER_LOAD_B64 high") ||
      !Expect(buffer_load_memory.StoreU8(0x2200u, 0x81u),
              "expected buffer SMEM test write for S_BUFFER_LOAD_I8") ||
      !Expect(buffer_load_memory.StoreU8(0x2201u, 0x7eu),
              "expected buffer SMEM test write for S_BUFFER_LOAD_U8") ||
      !Expect(buffer_load_memory.StoreU16(0x2202u, 0x8002u),
              "expected buffer SMEM test write for S_BUFFER_LOAD_I16") ||
      !Expect(buffer_load_memory.StoreU16(0x2204u, 0x8124u),
              "expected buffer SMEM test write for S_BUFFER_LOAD_U16")) {
    return 1;
  }
  for (std::size_t i = 0; i < expected_buffer_load_b96.size(); ++i) {
    if (!Expect(buffer_load_memory.WriteU32(
                    0x2050u + static_cast<std::uint32_t>(i * 4u),
                    expected_buffer_load_b96[i]),
                "expected buffer SMEM test write for S_BUFFER_LOAD_B96")) {
      return 1;
    }
  }
  for (std::size_t i = 0; i < expected_buffer_load_b128.size(); ++i) {
    if (!Expect(buffer_load_memory.WriteU32(
                    0x2078u + static_cast<std::uint32_t>(i * 4u),
                    expected_buffer_load_b128[i]),
                "expected buffer SMEM test write for S_BUFFER_LOAD_B128")) {
      return 1;
    }
  }
  for (std::size_t i = 0; i < expected_buffer_load_b256.size(); ++i) {
    if (!Expect(buffer_load_memory.WriteU32(
                    0x20a8u + static_cast<std::uint32_t>(i * 4u),
                    expected_buffer_load_b256[i]),
                "expected buffer SMEM test write for S_BUFFER_LOAD_B256")) {
      return 1;
    }
  }
  for (std::size_t i = 0; i < expected_buffer_load_b512.size(); ++i) {
    if (!Expect(buffer_load_memory.WriteU32(
                    0x2124u + static_cast<std::uint32_t>(i * 4u),
                    expected_buffer_load_b512[i]),
                "expected buffer SMEM test write for S_BUFFER_LOAD_B512")) {
      return 1;
    }
  }

  WaveExecutionState decoded_buffer_load_state;
  initialize_buffer_load_state(&decoded_buffer_load_state);
  if (!Expect(interpreter.ExecuteProgram(buffer_load_program,
                                         &decoded_buffer_load_state,
                                         &buffer_load_memory, &error_message),
              "expected decoded buffer SMEM load execution success") ||
      !Expect(expect_buffer_load_state(decoded_buffer_load_state),
              "expected decoded buffer SMEM load state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_buffer_load_program;
  if (!Expect(interpreter.CompileProgram(buffer_load_program,
                                         &compiled_buffer_load_program,
                                         &error_message),
              "expected compiled buffer SMEM load program success") ||
      !Expect(compiled_buffer_load_program.size() == 11u,
              "expected eleven compiled buffer SMEM load instructions") ||
      !Expect(compiled_buffer_load_program[0].opcode ==
                  Gfx1201CompiledOpcode::kSBufferLoadB32,
              "expected compiled S_BUFFER_LOAD_B32 opcode") ||
      !Expect(compiled_buffer_load_program[1].opcode ==
                  Gfx1201CompiledOpcode::kSBufferLoadB64,
              "expected compiled S_BUFFER_LOAD_B64 opcode") ||
      !Expect(compiled_buffer_load_program[2].opcode ==
                  Gfx1201CompiledOpcode::kSBufferLoadB96,
              "expected compiled S_BUFFER_LOAD_B96 opcode") ||
      !Expect(compiled_buffer_load_program[3].opcode ==
                  Gfx1201CompiledOpcode::kSBufferLoadB128,
              "expected compiled S_BUFFER_LOAD_B128 opcode") ||
      !Expect(compiled_buffer_load_program[4].opcode ==
                  Gfx1201CompiledOpcode::kSBufferLoadB256,
              "expected compiled S_BUFFER_LOAD_B256 opcode") ||
      !Expect(compiled_buffer_load_program[5].opcode ==
                  Gfx1201CompiledOpcode::kSBufferLoadB512,
              "expected compiled S_BUFFER_LOAD_B512 opcode") ||
      !Expect(compiled_buffer_load_program[6].opcode ==
                  Gfx1201CompiledOpcode::kSBufferLoadI8,
              "expected compiled S_BUFFER_LOAD_I8 opcode") ||
      !Expect(compiled_buffer_load_program[7].opcode ==
                  Gfx1201CompiledOpcode::kSBufferLoadU8,
              "expected compiled S_BUFFER_LOAD_U8 opcode") ||
      !Expect(compiled_buffer_load_program[8].opcode ==
                  Gfx1201CompiledOpcode::kSBufferLoadI16,
              "expected compiled S_BUFFER_LOAD_I16 opcode") ||
      !Expect(compiled_buffer_load_program[9].opcode ==
                  Gfx1201CompiledOpcode::kSBufferLoadU16,
              "expected compiled S_BUFFER_LOAD_U16 opcode") ||
      !Expect(compiled_buffer_load_program[10].opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after buffer SMEM loads")) {
    return 1;
  }

  WaveExecutionState compiled_buffer_load_state;
  initialize_buffer_load_state(&compiled_buffer_load_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_buffer_load_program,
                                         &compiled_buffer_load_state,
                                         &buffer_load_memory, &error_message),
              "expected compiled buffer SMEM load execution success") ||
      !Expect(expect_buffer_load_state(compiled_buffer_load_state),
              "expected compiled buffer SMEM load state")) {
    return 1;
  }

  const auto prefetch_inst_pc_rel_words =
      MakeSmemPrefetchPcRel(37u, -32, 9u, -3);
  const auto prefetch_data_pc_rel_words =
      MakeSmemPrefetchPcRel(40u, 48, 5u, 7);
  const auto prefetch_inst_words = MakeSmemBasePrefetch(36u, 8u, -16, 11u, -4);
  const auto prefetch_data_words = MakeSmemBasePrefetch(38u, 12u, 64, 7u, 3);
  const auto buffer_prefetch_words =
      MakeSmemBasePrefetch(39u, 20u, 24, 13u, -1);
  const auto atc_probe_words = MakeSmem(34u, 42u, 6u, false, 17u, true);
  const auto atc_probe_buffer_words = MakeSmem(35u, 55u, 10u, true, 0x1abcdu);
  DecodedInstruction prefetch_inst_pc_rel_instruction;
  std::size_t prefetch_words_consumed = 0;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      prefetch_inst_pc_rel_words.data(),
                      prefetch_inst_pc_rel_words.size()),
                  &prefetch_inst_pc_rel_instruction, &prefetch_words_consumed,
                  &error_message),
              "expected S_PREFETCH_INST_PC_REL direct decode success") ||
      !Expect(ExpectThreeOperandInstruction(
                  prefetch_inst_pc_rel_instruction, "S_PREFETCH_INST_PC_REL",
                  OperandKind::kImm32, static_cast<std::uint32_t>(-32),
                  OperandKind::kSgpr, 9u, OperandKind::kImm32,
                  static_cast<std::uint32_t>(-3)),
              "expected decoded S_PREFETCH_INST_PC_REL operands") ||
      !Expect(prefetch_words_consumed == 2u,
              "expected S_PREFETCH_INST_PC_REL to consume two dwords")) {
    return 1;
  }

  DecodedInstruction prefetch_data_pc_rel_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(
                      prefetch_data_pc_rel_words.data(),
                      prefetch_data_pc_rel_words.size()),
                  &prefetch_data_pc_rel_instruction, &prefetch_words_consumed,
                  &error_message),
              "expected S_PREFETCH_DATA_PC_REL direct decode success") ||
      !Expect(ExpectThreeOperandInstruction(
                  prefetch_data_pc_rel_instruction, "S_PREFETCH_DATA_PC_REL",
                  OperandKind::kImm32, 48u, OperandKind::kSgpr, 5u,
                  OperandKind::kImm32, 7u),
              "expected decoded S_PREFETCH_DATA_PC_REL operands") ||
      !Expect(prefetch_words_consumed == 2u,
              "expected S_PREFETCH_DATA_PC_REL to consume two dwords")) {
    return 1;
  }

  DecodedInstruction prefetch_inst_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(prefetch_inst_words.data(),
                                                 prefetch_inst_words.size()),
                  &prefetch_inst_instruction, &prefetch_words_consumed,
                  &error_message),
              "expected S_PREFETCH_INST direct decode success") ||
      !Expect(ExpectFourOperandInstruction(
                  prefetch_inst_instruction, "S_PREFETCH_INST",
                  OperandKind::kSgpr, 8u, OperandKind::kImm32,
                  static_cast<std::uint32_t>(-16), OperandKind::kSgpr, 11u,
                  OperandKind::kImm32, static_cast<std::uint32_t>(-4)),
              "expected decoded S_PREFETCH_INST operands") ||
      !Expect(prefetch_words_consumed == 2u,
              "expected S_PREFETCH_INST to consume two dwords") ||
      !Expect(ExpectOperandDescriptor(
                  prefetch_inst_instruction.operands[0],
                  OperandRole::kSource0, OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 64u, 2u, false),
              "expected S_PREFETCH_INST base descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  prefetch_inst_instruction.operands[1],
                  OperandRole::kSource1, OperandSlotKind::kSource1,
                  OperandValueClass::kUnknown, OperandAccess::kRead,
                  FragmentKind::kScalar, 32u, 1u, false),
              "expected S_PREFETCH_INST ioffset descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  prefetch_inst_instruction.operands[2],
                  OperandRole::kSource2, OperandSlotKind::kSource2,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 32u, 1u, false),
              "expected S_PREFETCH_INST soffset descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  prefetch_inst_instruction.operands[3],
                  OperandRole::kUnknown, OperandSlotKind::kUnknown,
                  OperandValueClass::kUnknown, OperandAccess::kRead,
                  FragmentKind::kScalar, 32u, 1u, false),
              "expected S_PREFETCH_INST sdata descriptor")) {
    return 1;
  }

  DecodedInstruction prefetch_data_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(prefetch_data_words.data(),
                                                 prefetch_data_words.size()),
                  &prefetch_data_instruction, &prefetch_words_consumed,
                  &error_message),
              "expected S_PREFETCH_DATA direct decode success") ||
      !Expect(ExpectFourOperandInstruction(
                  prefetch_data_instruction, "S_PREFETCH_DATA",
                  OperandKind::kSgpr, 12u, OperandKind::kImm32, 64u,
                  OperandKind::kSgpr, 7u, OperandKind::kImm32, 3u),
              "expected decoded S_PREFETCH_DATA operands") ||
      !Expect(prefetch_words_consumed == 2u,
              "expected S_PREFETCH_DATA to consume two dwords")) {
    return 1;
  }

  DecodedInstruction buffer_prefetch_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(buffer_prefetch_words.data(),
                                                 buffer_prefetch_words.size()),
                  &buffer_prefetch_instruction, &prefetch_words_consumed,
                  &error_message),
              "expected S_BUFFER_PREFETCH_DATA direct decode success") ||
      !Expect(ExpectFourOperandInstruction(
                  buffer_prefetch_instruction, "S_BUFFER_PREFETCH_DATA",
                  OperandKind::kSgpr, 20u, OperandKind::kImm32, 24u,
                  OperandKind::kSgpr, 13u, OperandKind::kImm32,
                  static_cast<std::uint32_t>(-1)),
              "expected decoded S_BUFFER_PREFETCH_DATA operands") ||
      !Expect(prefetch_words_consumed == 2u,
              "expected S_BUFFER_PREFETCH_DATA to consume two dwords") ||
      !Expect(ExpectOperandDescriptor(
                  buffer_prefetch_instruction.operands[0],
                  OperandRole::kSource0, OperandSlotKind::kSource0,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 128u, 4u, false),
              "expected S_BUFFER_PREFETCH_DATA base descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  buffer_prefetch_instruction.operands[3],
                  OperandRole::kUnknown, OperandSlotKind::kUnknown,
                  OperandValueClass::kUnknown, OperandAccess::kRead,
                  FragmentKind::kScalar, 32u, 1u, false),
              "expected S_BUFFER_PREFETCH_DATA sdata descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 5> prefetch_pc_rel_program_words{
      prefetch_inst_pc_rel_words[0], prefetch_inst_pc_rel_words[1],
      prefetch_data_pc_rel_words[0], prefetch_data_pc_rel_words[1],
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> prefetch_pc_rel_program;
  if (!Expect(decoder.DecodeProgram(prefetch_pc_rel_program_words,
                                    &prefetch_pc_rel_program, &error_message),
              "expected PC-relative prefetch program decode success") ||
      !Expect(prefetch_pc_rel_program.size() == 3u,
              "expected three decoded PC-relative prefetch program "
              "instructions") ||
      !Expect(prefetch_pc_rel_program[0].opcode == "S_PREFETCH_INST_PC_REL",
              "expected decoded S_PREFETCH_INST_PC_REL program opcode") ||
      !Expect(prefetch_pc_rel_program[1].opcode == "S_PREFETCH_DATA_PC_REL",
              "expected decoded S_PREFETCH_DATA_PC_REL program opcode") ||
      !Expect(prefetch_pc_rel_program[2].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after PC-relative prefetch ops")) {
    return 1;
  }

  auto initialize_prefetch_pc_rel_state = [](WaveExecutionState* state) {
    state->exec_mask = 0x9u;
    state->sgprs[5] = 0x11111111u;
    state->sgprs[9] = 0x22222222u;
    state->vgprs[7][0] = 0xabcdef01u;
  };
  auto expect_prefetch_pc_rel_state = [](const WaveExecutionState& state) {
    return state.lane_count == 32u && state.exec_mask == 0x9u &&
           state.sgprs[5] == 0x11111111u &&
           state.sgprs[9] == 0x22222222u &&
           state.vgprs[7][0] == 0xabcdef01u && state.halted &&
           !state.waiting_on_barrier && state.pc == 2u;
  };

  WaveExecutionState decoded_prefetch_pc_rel_state;
  initialize_prefetch_pc_rel_state(&decoded_prefetch_pc_rel_state);
  if (!Expect(interpreter.ExecuteProgram(prefetch_pc_rel_program,
                                         &decoded_prefetch_pc_rel_state,
                                         &error_message),
              "expected decoded PC-relative prefetch execution success") ||
      !Expect(expect_prefetch_pc_rel_state(decoded_prefetch_pc_rel_state),
              "expected decoded PC-relative prefetch state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_prefetch_pc_rel_program;
  if (!Expect(interpreter.CompileProgram(prefetch_pc_rel_program,
                                         &compiled_prefetch_pc_rel_program,
                                         &error_message),
              "expected compiled PC-relative prefetch program success") ||
      !Expect(compiled_prefetch_pc_rel_program.size() == 3u,
              "expected three compiled PC-relative prefetch program "
              "instructions") ||
      !Expect(compiled_prefetch_pc_rel_program[0].opcode ==
                  Gfx1201CompiledOpcode::kSNop,
              "expected compiled S_PREFETCH_INST_PC_REL opcode") ||
      !Expect(compiled_prefetch_pc_rel_program[1].opcode ==
                  Gfx1201CompiledOpcode::kSNop,
              "expected compiled S_PREFETCH_DATA_PC_REL opcode") ||
      !Expect(compiled_prefetch_pc_rel_program[2].opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after PC-relative prefetch ops")) {
    return 1;
  }

  WaveExecutionState compiled_prefetch_pc_rel_state;
  initialize_prefetch_pc_rel_state(&compiled_prefetch_pc_rel_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_prefetch_pc_rel_program,
                                         &compiled_prefetch_pc_rel_state,
                                         &error_message),
              "expected compiled PC-relative prefetch execution success") ||
      !Expect(expect_prefetch_pc_rel_state(compiled_prefetch_pc_rel_state),
              "expected compiled PC-relative prefetch state")) {
    return 1;
  }

  const std::array<std::uint32_t, 7> prefetch_base_program_words{
      prefetch_inst_words[0], prefetch_inst_words[1], prefetch_data_words[0],
      prefetch_data_words[1], buffer_prefetch_words[0], buffer_prefetch_words[1],
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> prefetch_base_program;
  if (!Expect(decoder.DecodeProgram(prefetch_base_program_words,
                                    &prefetch_base_program, &error_message),
              "expected base prefetch program decode success") ||
      !Expect(prefetch_base_program.size() == 4u,
              "expected four decoded base prefetch program instructions") ||
      !Expect(prefetch_base_program[0].opcode == "S_PREFETCH_INST",
              "expected decoded S_PREFETCH_INST program opcode") ||
      !Expect(prefetch_base_program[1].opcode == "S_PREFETCH_DATA",
              "expected decoded S_PREFETCH_DATA program opcode") ||
      !Expect(prefetch_base_program[2].opcode == "S_BUFFER_PREFETCH_DATA",
              "expected decoded S_BUFFER_PREFETCH_DATA program opcode") ||
      !Expect(prefetch_base_program[3].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after base prefetch ops")) {
    return 1;
  }

  auto initialize_prefetch_base_state = [](WaveExecutionState* state) {
    state->exec_mask = 0x15u;
    state->sgprs[7] = 0x01010101u;
    state->sgprs[8] = 0x11111111u;
    state->sgprs[9] = 0x22222222u;
    state->sgprs[11] = 0x33333333u;
    state->sgprs[12] = 0x44444444u;
    state->sgprs[13] = 0x55555555u;
    state->sgprs[20] = 0x66666666u;
    state->sgprs[21] = 0x77777777u;
    state->sgprs[22] = 0x88888888u;
    state->sgprs[23] = 0x99999999u;
    state->vgprs[9][0] = 0xabcdef01u;
  };
  auto expect_prefetch_base_state = [](const WaveExecutionState& state) {
    return state.lane_count == 32u && state.exec_mask == 0x15u &&
           state.sgprs[7] == 0x01010101u &&
           state.sgprs[8] == 0x11111111u &&
           state.sgprs[9] == 0x22222222u &&
           state.sgprs[11] == 0x33333333u &&
           state.sgprs[12] == 0x44444444u &&
           state.sgprs[13] == 0x55555555u &&
           state.sgprs[20] == 0x66666666u &&
           state.sgprs[21] == 0x77777777u &&
           state.sgprs[22] == 0x88888888u &&
           state.sgprs[23] == 0x99999999u &&
           state.vgprs[9][0] == 0xabcdef01u && state.halted &&
           !state.waiting_on_barrier && state.pc == 3u;
  };

  WaveExecutionState decoded_prefetch_base_state;
  initialize_prefetch_base_state(&decoded_prefetch_base_state);
  if (!Expect(interpreter.ExecuteProgram(prefetch_base_program,
                                         &decoded_prefetch_base_state,
                                         &error_message),
              "expected decoded base prefetch execution success") ||
      !Expect(expect_prefetch_base_state(decoded_prefetch_base_state),
              "expected decoded base prefetch state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_prefetch_base_program;
  if (!Expect(interpreter.CompileProgram(prefetch_base_program,
                                         &compiled_prefetch_base_program,
                                         &error_message),
              "expected compiled base prefetch program success") ||
      !Expect(compiled_prefetch_base_program.size() == 4u,
              "expected four compiled base prefetch program instructions") ||
      !Expect(compiled_prefetch_base_program[0].opcode ==
                  Gfx1201CompiledOpcode::kSNop,
              "expected compiled S_PREFETCH_INST opcode") ||
      !Expect(compiled_prefetch_base_program[1].opcode ==
                  Gfx1201CompiledOpcode::kSNop,
              "expected compiled S_PREFETCH_DATA opcode") ||
      !Expect(compiled_prefetch_base_program[2].opcode ==
                  Gfx1201CompiledOpcode::kSNop,
              "expected compiled S_BUFFER_PREFETCH_DATA opcode") ||
      !Expect(compiled_prefetch_base_program[3].opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after base prefetch ops")) {
    return 1;
  }

  WaveExecutionState compiled_prefetch_base_state;
  initialize_prefetch_base_state(&compiled_prefetch_base_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_prefetch_base_program,
                                         &compiled_prefetch_base_state,
                                         &error_message),
              "expected compiled base prefetch execution success") ||
      !Expect(expect_prefetch_base_state(compiled_prefetch_base_state),
              "expected compiled base prefetch state")) {
    return 1;
  }

  DecodedInstruction atc_probe_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(atc_probe_words.data(),
                                                 atc_probe_words.size()),
                  &atc_probe_instruction, &prefetch_words_consumed,
                  &error_message),
              "expected S_ATC_PROBE direct decode success") ||
      !Expect(ExpectThreeOperandInstruction(atc_probe_instruction,
                                            "S_ATC_PROBE", OperandKind::kImm32,
                                            42u, OperandKind::kSgpr, 6u,
                                            OperandKind::kSgpr, 17u),
              "expected decoded S_ATC_PROBE operands") ||
      !Expect(prefetch_words_consumed == 2u,
              "expected S_ATC_PROBE to consume two dwords") ||
      !Expect(ExpectOperandDescriptor(
                  atc_probe_instruction.operands[0], OperandRole::kSource0,
                  OperandSlotKind::kSource0, OperandValueClass::kUnknown,
                  OperandAccess::kRead, FragmentKind::kScalar, 8u, 1u, false),
              "expected S_ATC_PROBE sdata descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  atc_probe_instruction.operands[1], OperandRole::kSource1,
                  OperandSlotKind::kSource1,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 64u, 2u, false),
              "expected S_ATC_PROBE base descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  atc_probe_instruction.operands[2], OperandRole::kSource2,
                  OperandSlotKind::kSource2,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 32u, 1u, false),
              "expected S_ATC_PROBE offset descriptor")) {
    return 1;
  }

  DecodedInstruction atc_probe_buffer_instruction;
  if (!Expect(decoder.DecodeInstruction(
                  std::span<const std::uint32_t>(atc_probe_buffer_words.data(),
                                                 atc_probe_buffer_words.size()),
                  &atc_probe_buffer_instruction, &prefetch_words_consumed,
                  &error_message),
              "expected S_ATC_PROBE_BUFFER direct decode success") ||
      !Expect(ExpectThreeOperandInstruction(
                  atc_probe_buffer_instruction, "S_ATC_PROBE_BUFFER",
                  OperandKind::kImm32, 55u, OperandKind::kSgpr, 10u,
                  OperandKind::kImm32, 0x1abcdu),
              "expected decoded S_ATC_PROBE_BUFFER operands") ||
      !Expect(prefetch_words_consumed == 2u,
              "expected S_ATC_PROBE_BUFFER to consume two dwords") ||
      !Expect(ExpectOperandDescriptor(
                  atc_probe_buffer_instruction.operands[0],
                  OperandRole::kSource0, OperandSlotKind::kSource0,
                  OperandValueClass::kUnknown, OperandAccess::kRead,
                  FragmentKind::kScalar, 8u, 1u, false),
              "expected S_ATC_PROBE_BUFFER sdata descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  atc_probe_buffer_instruction.operands[1],
                  OperandRole::kSource1, OperandSlotKind::kSource1,
                  OperandValueClass::kScalarRegister, OperandAccess::kRead,
                  FragmentKind::kScalar, 128u, 4u, false),
              "expected S_ATC_PROBE_BUFFER base descriptor") ||
      !Expect(ExpectOperandDescriptor(
                  atc_probe_buffer_instruction.operands[2],
                  OperandRole::kSource2, OperandSlotKind::kSource2,
                  OperandValueClass::kUnknown, OperandAccess::kRead,
                  FragmentKind::kScalar, 32u, 1u, false),
              "expected S_ATC_PROBE_BUFFER offset descriptor")) {
    return 1;
  }

  const std::array<std::uint32_t, 5> atc_probe_program_words{
      atc_probe_words[0],        atc_probe_words[1],
      atc_probe_buffer_words[0], atc_probe_buffer_words[1],
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> atc_probe_program;
  if (!Expect(decoder.DecodeProgram(atc_probe_program_words, &atc_probe_program,
                                    &error_message),
              "expected ATC probe program decode success") ||
      !Expect(atc_probe_program.size() == 3u,
              "expected three decoded ATC probe program instructions") ||
      !Expect(atc_probe_program[0].opcode == "S_ATC_PROBE",
              "expected decoded S_ATC_PROBE program opcode") ||
      !Expect(atc_probe_program[1].opcode == "S_ATC_PROBE_BUFFER",
              "expected decoded S_ATC_PROBE_BUFFER program opcode") ||
      !Expect(atc_probe_program[2].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after ATC probe ops")) {
    return 1;
  }

  auto initialize_atc_probe_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xdu;
    state->sgprs[6] = 0x11111111u;
    state->sgprs[7] = 0x22222222u;
    state->sgprs[17] = 0x33333333u;
    state->sgprs[10] = 0x44444444u;
    state->sgprs[11] = 0x55555555u;
    state->sgprs[12] = 0x66666666u;
    state->sgprs[13] = 0x77777777u;
    state->vgprs[4][0] = 0x12345678u;
  };
  auto expect_atc_probe_state = [](const WaveExecutionState& state) {
    return state.lane_count == 32u && state.exec_mask == 0xdu &&
           state.sgprs[6] == 0x11111111u &&
           state.sgprs[7] == 0x22222222u &&
           state.sgprs[17] == 0x33333333u &&
           state.sgprs[10] == 0x44444444u &&
           state.sgprs[11] == 0x55555555u &&
           state.sgprs[12] == 0x66666666u &&
           state.sgprs[13] == 0x77777777u &&
           state.vgprs[4][0] == 0x12345678u && state.halted &&
           !state.waiting_on_barrier && state.pc == 2u;
  };

  WaveExecutionState decoded_atc_probe_state;
  initialize_atc_probe_state(&decoded_atc_probe_state);
  if (!Expect(interpreter.ExecuteProgram(atc_probe_program,
                                         &decoded_atc_probe_state,
                                         &error_message),
              "expected decoded ATC probe execution success") ||
      !Expect(expect_atc_probe_state(decoded_atc_probe_state),
              "expected decoded ATC probe state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_atc_probe_program;
  if (!Expect(interpreter.CompileProgram(atc_probe_program,
                                         &compiled_atc_probe_program,
                                         &error_message),
              "expected compiled ATC probe program success") ||
      !Expect(compiled_atc_probe_program.size() == 3u,
              "expected three compiled ATC probe program instructions") ||
      !Expect(compiled_atc_probe_program[0].opcode ==
                  Gfx1201CompiledOpcode::kSNop,
              "expected compiled S_ATC_PROBE opcode") ||
      !Expect(compiled_atc_probe_program[1].opcode ==
                  Gfx1201CompiledOpcode::kSNop,
              "expected compiled S_ATC_PROBE_BUFFER opcode") ||
      !Expect(compiled_atc_probe_program[2].opcode ==
                  Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM after ATC probe ops")) {
    return 1;
  }

  WaveExecutionState compiled_atc_probe_state;
  initialize_atc_probe_state(&compiled_atc_probe_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_atc_probe_program,
                                         &compiled_atc_probe_state,
                                         &error_message),
              "expected compiled ATC probe execution success") ||
      !Expect(expect_atc_probe_state(compiled_atc_probe_state),
              "expected compiled ATC probe state")) {
    return 1;
  }

  const std::array<std::uint32_t, 2> wide_shift_words{
      MakeVop2(31u, 124u, 263u, 8u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> wide_shift_program;
  if (!Expect(decoder.DecodeProgram(wide_shift_words, &wide_shift_program,
                                    &error_message),
              "expected wide shift program decode success") ||
      !Expect(wide_shift_program.size() == 2u,
              "expected two decoded wide shift instructions") ||
      !Expect(wide_shift_program[0].opcode == "V_LSHLREV_B64",
              "expected decoded V_LSHLREV_B64") ||
      !Expect(wide_shift_program[1].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM after wide shift batch")) {
    return 1;
  }

  auto initialize_wide_shift_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xbu;

    state->vgprs[7][0] = 4u;
    state->vgprs[7][1] = 8u;
    state->vgprs[7][2] = 0x77777777u;
    state->vgprs[7][3] = 40u;

    SplitU64(0x3ull, &state->vgprs[8][0], &state->vgprs[9][0]);
    SplitU64(0x1ull, &state->vgprs[8][1], &state->vgprs[9][1]);
    state->vgprs[8][2] = 0x88888888u;
    state->vgprs[9][2] = 0x99999999u;
    SplitU64(0x1ull, &state->vgprs[8][3], &state->vgprs[9][3]);

    state->vgprs[124][2] = 0xe4e4e4e4u;
    state->vgprs[125][2] = 0xf5f5f5f5u;
  };

  WaveExecutionState decoded_wide_shift_state;
  initialize_wide_shift_state(&decoded_wide_shift_state);
  if (!Expect(interpreter.ExecuteProgram(wide_shift_program,
                                         &decoded_wide_shift_state,
                                         &error_message),
              "expected decoded wide shift execution success") ||
      !Expect(ExpectWideShiftSeedState(decoded_wide_shift_state),
              "expected decoded wide shift state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_wide_shift_program;
  if (!Expect(interpreter.CompileProgram(wide_shift_program,
                                         &compiled_wide_shift_program,
                                         &error_message),
              "expected compiled wide shift program success") ||
      !Expect(compiled_wide_shift_program.size() == 2u,
              "expected two compiled wide shift instructions") ||
      !Expect(compiled_wide_shift_program[0].opcode ==
                  Gfx1201CompiledOpcode::kVLshlrevB64,
              "expected compiled V_LSHLREV_B64 opcode")) {
    return 1;
  }

  WaveExecutionState compiled_wide_shift_state;
  initialize_wide_shift_state(&compiled_wide_shift_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_wide_shift_program,
                                         &compiled_wide_shift_state,
                                         &error_message),
              "expected compiled wide shift execution success") ||
      !Expect(ExpectWideShiftSeedState(compiled_wide_shift_state),
              "expected compiled wide shift state")) {
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

  struct RemainingI16U16CompareCase {
    const char* opcode;
    std::uint32_t encoded_opcode;
    std::uint64_t expected_mask;
    Gfx1201CompiledOpcode compiled_opcode;
    bool writes_exec;
    bool is_signed;
  };

  constexpr std::array<RemainingI16U16CompareCase, 24>
      kRemainingI16U16CompareCases{{
          {"V_CMP_EQ_I16", 50u, 2u, Gfx1201CompiledOpcode::kVCmpEqI16, false,
           true},
          {"V_CMP_NE_I16", 53u, 13u, Gfx1201CompiledOpcode::kVCmpNeI16, false,
           true},
          {"V_CMP_LT_I16", 49u, 12u, Gfx1201CompiledOpcode::kVCmpLtI16, false,
           true},
          {"V_CMP_LE_I16", 51u, 14u, Gfx1201CompiledOpcode::kVCmpLeI16, false,
           true},
          {"V_CMP_GT_I16", 52u, 1u, Gfx1201CompiledOpcode::kVCmpGtI16, false,
           true},
          {"V_CMP_GE_I16", 54u, 3u, Gfx1201CompiledOpcode::kVCmpGeI16, false,
           true},
          {"V_CMP_EQ_U16", 58u, 2u, Gfx1201CompiledOpcode::kVCmpEqU16, false,
           false},
          {"V_CMP_NE_U16", 61u, 13u, Gfx1201CompiledOpcode::kVCmpNeU16, false,
           false},
          {"V_CMP_LT_U16", 57u, 12u, Gfx1201CompiledOpcode::kVCmpLtU16, false,
           false},
          {"V_CMP_LE_U16", 59u, 14u, Gfx1201CompiledOpcode::kVCmpLeU16, false,
           false},
          {"V_CMP_GT_U16", 60u, 1u, Gfx1201CompiledOpcode::kVCmpGtU16, false,
           false},
          {"V_CMP_GE_U16", 62u, 3u, Gfx1201CompiledOpcode::kVCmpGeU16, false,
           false},
          {"V_CMPX_EQ_I16", 178u, 2u, Gfx1201CompiledOpcode::kVCmpxEqI16, true,
           true},
          {"V_CMPX_NE_I16", 181u, 13u, Gfx1201CompiledOpcode::kVCmpxNeI16,
           true, true},
          {"V_CMPX_LT_I16", 177u, 12u, Gfx1201CompiledOpcode::kVCmpxLtI16,
           true, true},
          {"V_CMPX_LE_I16", 179u, 14u, Gfx1201CompiledOpcode::kVCmpxLeI16,
           true, true},
          {"V_CMPX_GT_I16", 180u, 1u, Gfx1201CompiledOpcode::kVCmpxGtI16, true,
           true},
          {"V_CMPX_GE_I16", 182u, 3u, Gfx1201CompiledOpcode::kVCmpxGeI16, true,
           true},
          {"V_CMPX_EQ_U16", 186u, 2u, Gfx1201CompiledOpcode::kVCmpxEqU16, true,
           false},
          {"V_CMPX_NE_U16", 189u, 13u, Gfx1201CompiledOpcode::kVCmpxNeU16,
           true, false},
          {"V_CMPX_LT_U16", 185u, 12u, Gfx1201CompiledOpcode::kVCmpxLtU16,
           true, false},
          {"V_CMPX_LE_U16", 187u, 14u, Gfx1201CompiledOpcode::kVCmpxLeU16,
           true, false},
          {"V_CMPX_GT_U16", 188u, 1u, Gfx1201CompiledOpcode::kVCmpxGtU16, true,
           false},
          {"V_CMPX_GE_U16", 190u, 3u, Gfx1201CompiledOpcode::kVCmpxGeU16, true,
           false},
      }};

  constexpr std::uint32_t kI16CompareLhsSimm16 = 0xfffeu;
  constexpr std::uint32_t kU16CompareLhsSimm16 = 0x0002u;
  constexpr std::uint32_t kI16RhsLane0 = 0xfffdu;
  constexpr std::uint32_t kI16RhsLane1 = 0xfffeu;
  constexpr std::uint32_t kI16RhsLane2 = 0x0001u;
  constexpr std::uint32_t kI16RhsLane3 = 0xffffu;
  constexpr std::uint32_t kU16RhsLane0 = 0x0001u;
  constexpr std::uint32_t kU16RhsLane1 = 0x0002u;
  constexpr std::uint32_t kU16RhsLane2 = 0x0003u;
  constexpr std::uint32_t kU16RhsLane3 = 0xffffu;

  for (const RemainingI16U16CompareCase& test_case :
       kRemainingI16U16CompareCases) {
    const std::array<std::uint32_t, 3> remaining_i16_u16_compare_words{
        MakeSopk(0u, 122u,
                 test_case.is_signed ? kI16CompareLhsSimm16
                                     : kU16CompareLhsSimm16),
        MakeVopc(test_case.encoded_opcode, 122u, 104u),
        MakeSopp(48u),
    };
    std::vector<DecodedInstruction> remaining_i16_u16_compare_program;
    if (!Expect(decoder.DecodeProgram(remaining_i16_u16_compare_words,
                                      &remaining_i16_u16_compare_program,
                                      &error_message),
                "expected remaining I16/U16 compare program decode success") ||
        !Expect(remaining_i16_u16_compare_program.size() == 3u,
                "expected three decoded remaining I16/U16 compare instructions")
        ||
        !Expect(remaining_i16_u16_compare_program[1].opcode == test_case.opcode,
                "expected decoded remaining I16/U16 compare opcode")) {
      return 1;
    }

    auto initialize_i16_u16_compare_state = [&](WaveExecutionState* state) {
      state->exec_mask = 0xfu;
      state->vcc_mask = 0x80u;
      state->vgprs[104][0] =
          test_case.is_signed ? kI16RhsLane0 : kU16RhsLane0;
      state->vgprs[104][1] =
          test_case.is_signed ? kI16RhsLane1 : kU16RhsLane1;
      state->vgprs[104][2] =
          test_case.is_signed ? kI16RhsLane2 : kU16RhsLane2;
      state->vgprs[104][3] =
          test_case.is_signed ? kI16RhsLane3 : kU16RhsLane3;
    };

    WaveExecutionState decoded_remaining_i16_u16_compare_state;
    initialize_i16_u16_compare_state(&decoded_remaining_i16_u16_compare_state);
    if (!Expect(interpreter.ExecuteProgram(
                    remaining_i16_u16_compare_program,
                    &decoded_remaining_i16_u16_compare_state, &error_message),
                "expected decoded remaining I16/U16 compare execution success")
        ||
        !Expect(decoded_remaining_i16_u16_compare_state.vcc_mask ==
                    (test_case.writes_exec
                         ? test_case.expected_mask
                         : (test_case.expected_mask | 0x80u)),
                "expected remaining I16/U16 compare VCC mask") ||
        !Expect(decoded_remaining_i16_u16_compare_state.exec_mask ==
                    (test_case.writes_exec ? test_case.expected_mask : 0xfu),
                "expected remaining I16/U16 compare EXEC mask") ||
        !Expect(decoded_remaining_i16_u16_compare_state.halted,
                "expected remaining I16/U16 compare program to halt") ||
        !Expect(decoded_remaining_i16_u16_compare_state.pc == 2u,
                "expected remaining I16/U16 compare program advance")) {
      return 1;
    }

    std::vector<Gfx1201CompiledInstruction>
        compiled_remaining_i16_u16_compare_program;
    if (!Expect(interpreter.CompileProgram(
                    remaining_i16_u16_compare_program,
                    &compiled_remaining_i16_u16_compare_program,
                    &error_message),
                "expected compiled remaining I16/U16 compare program success")
        ||
        !Expect(compiled_remaining_i16_u16_compare_program[1].opcode ==
                    test_case.compiled_opcode,
                "expected compiled remaining I16/U16 compare opcode")) {
      return 1;
    }

    WaveExecutionState compiled_remaining_i16_u16_compare_state;
    initialize_i16_u16_compare_state(&compiled_remaining_i16_u16_compare_state);
    if (!Expect(interpreter.ExecuteProgram(
                    compiled_remaining_i16_u16_compare_program,
                    &compiled_remaining_i16_u16_compare_state, &error_message),
                "expected compiled remaining I16/U16 compare execution success")
        ||
        !Expect(compiled_remaining_i16_u16_compare_state.vcc_mask ==
                    (test_case.writes_exec
                         ? test_case.expected_mask
                         : (test_case.expected_mask | 0x80u)),
                "expected compiled remaining I16/U16 compare VCC mask") ||
        !Expect(compiled_remaining_i16_u16_compare_state.exec_mask ==
                    (test_case.writes_exec ? test_case.expected_mask : 0xfu),
                "expected compiled remaining I16/U16 compare EXEC mask") ||
        !Expect(compiled_remaining_i16_u16_compare_state.halted,
                "expected compiled remaining I16/U16 compare program to halt")
        ||
        !Expect(compiled_remaining_i16_u16_compare_state.pc == 2u,
                "expected compiled remaining I16/U16 compare program advance")) {
      return 1;
    }
  }

  struct RemainingF16CompareCase {
    const char* opcode;
    std::uint32_t encoded_opcode;
    std::uint64_t expected_mask;
    Gfx1201CompiledOpcode compiled_opcode;
    bool writes_exec;
  };

  constexpr std::array<RemainingF16CompareCase, 28> kRemainingF16CompareCases{{
      {"V_CMP_EQ_F16", 2u, 2u, Gfx1201CompiledOpcode::kVCmpEqF16, false},
      {"V_CMP_GE_F16", 6u, 3u, Gfx1201CompiledOpcode::kVCmpGeF16, false},
      {"V_CMP_GT_F16", 4u, 1u, Gfx1201CompiledOpcode::kVCmpGtF16, false},
      {"V_CMP_LE_F16", 3u, 6u, Gfx1201CompiledOpcode::kVCmpLeF16, false},
      {"V_CMP_LG_F16", 5u, 5u, Gfx1201CompiledOpcode::kVCmpLgF16, false},
      {"V_CMP_LT_F16", 1u, 4u, Gfx1201CompiledOpcode::kVCmpLtF16, false},
      {"V_CMP_NEQ_F16", 13u, 13u, Gfx1201CompiledOpcode::kVCmpNeqF16, false},
      {"V_CMP_O_F16", 7u, 7u, Gfx1201CompiledOpcode::kVCmpOF16, false},
      {"V_CMP_U_F16", 8u, 8u, Gfx1201CompiledOpcode::kVCmpUF16, false},
      {"V_CMP_NGE_F16", 9u, 12u, Gfx1201CompiledOpcode::kVCmpNgeF16, false},
      {"V_CMP_NLG_F16", 10u, 10u, Gfx1201CompiledOpcode::kVCmpNlgF16, false},
      {"V_CMP_NGT_F16", 11u, 14u, Gfx1201CompiledOpcode::kVCmpNgtF16, false},
      {"V_CMP_NLE_F16", 12u, 9u, Gfx1201CompiledOpcode::kVCmpNleF16, false},
      {"V_CMP_NLT_F16", 14u, 11u, Gfx1201CompiledOpcode::kVCmpNltF16, false},
      {"V_CMPX_EQ_F16", 130u, 2u, Gfx1201CompiledOpcode::kVCmpxEqF16, true},
      {"V_CMPX_GE_F16", 134u, 3u, Gfx1201CompiledOpcode::kVCmpxGeF16, true},
      {"V_CMPX_GT_F16", 132u, 1u, Gfx1201CompiledOpcode::kVCmpxGtF16, true},
      {"V_CMPX_LE_F16", 131u, 6u, Gfx1201CompiledOpcode::kVCmpxLeF16, true},
      {"V_CMPX_LG_F16", 133u, 5u, Gfx1201CompiledOpcode::kVCmpxLgF16, true},
      {"V_CMPX_LT_F16", 129u, 4u, Gfx1201CompiledOpcode::kVCmpxLtF16, true},
      {"V_CMPX_NEQ_F16", 141u, 13u, Gfx1201CompiledOpcode::kVCmpxNeqF16,
       true},
      {"V_CMPX_O_F16", 135u, 7u, Gfx1201CompiledOpcode::kVCmpxOF16, true},
      {"V_CMPX_U_F16", 136u, 8u, Gfx1201CompiledOpcode::kVCmpxUF16, true},
      {"V_CMPX_NGE_F16", 137u, 12u, Gfx1201CompiledOpcode::kVCmpxNgeF16,
       true},
      {"V_CMPX_NLG_F16", 138u, 10u, Gfx1201CompiledOpcode::kVCmpxNlgF16,
       true},
      {"V_CMPX_NGT_F16", 139u, 14u, Gfx1201CompiledOpcode::kVCmpxNgtF16,
       true},
      {"V_CMPX_NLE_F16", 140u, 9u, Gfx1201CompiledOpcode::kVCmpxNleF16, true},
      {"V_CMPX_NLT_F16", 142u, 11u, Gfx1201CompiledOpcode::kVCmpxNltF16,
       true},
  }};

  constexpr std::uint32_t kF16CompareLhsBits = 0x4000u;
  constexpr std::uint32_t kF16CompareRhsLane0 = 0x3c00u;
  constexpr std::uint32_t kF16CompareRhsLane1 = 0x4000u;
  constexpr std::uint32_t kF16CompareRhsLane2 = 0x4200u;
  constexpr std::uint32_t kF16CompareRhsLane3 = kQuietNaNF16Bits;

  for (const RemainingF16CompareCase& test_case : kRemainingF16CompareCases) {
    const std::array<std::uint32_t, 3> remaining_f16_compare_words{
        MakeSopk(0u, 118u, kF16CompareLhsBits),
        MakeVopc(test_case.encoded_opcode, 118u, 94u),
        MakeSopp(48u),
    };
    std::vector<DecodedInstruction> remaining_f16_compare_program;
    if (!Expect(decoder.DecodeProgram(remaining_f16_compare_words,
                                      &remaining_f16_compare_program,
                                      &error_message),
                "expected remaining F16 compare program decode success") ||
        !Expect(remaining_f16_compare_program.size() == 3u,
                "expected three decoded remaining F16 compare instructions") ||
        !Expect(remaining_f16_compare_program[1].opcode == test_case.opcode,
                "expected decoded remaining F16 compare opcode")) {
      return 1;
    }

    auto initialize_f16_compare_state = [](WaveExecutionState* state) {
      state->exec_mask = 0xfu;
      state->vcc_mask = 0x80u;
      state->vgprs[94][0] = kF16CompareRhsLane0;
      state->vgprs[94][1] = kF16CompareRhsLane1;
      state->vgprs[94][2] = kF16CompareRhsLane2;
      state->vgprs[94][3] = kF16CompareRhsLane3;
    };

    WaveExecutionState decoded_remaining_f16_compare_state;
    initialize_f16_compare_state(&decoded_remaining_f16_compare_state);
    if (!Expect(interpreter.ExecuteProgram(remaining_f16_compare_program,
                                           &decoded_remaining_f16_compare_state,
                                           &error_message),
                "expected decoded remaining F16 compare execution success") ||
        !Expect(decoded_remaining_f16_compare_state.vcc_mask ==
                    (test_case.writes_exec
                         ? test_case.expected_mask
                         : (test_case.expected_mask | 0x80u)),
                "expected remaining F16 compare VCC mask") ||
        !Expect(decoded_remaining_f16_compare_state.exec_mask ==
                    (test_case.writes_exec ? test_case.expected_mask : 0xfu),
                "expected remaining F16 compare EXEC mask") ||
        !Expect(decoded_remaining_f16_compare_state.halted,
                "expected remaining F16 compare program to halt") ||
        !Expect(decoded_remaining_f16_compare_state.pc == 2u,
                "expected remaining F16 compare program advance")) {
      return 1;
    }

    std::vector<Gfx1201CompiledInstruction>
        compiled_remaining_f16_compare_program;
    if (!Expect(interpreter.CompileProgram(remaining_f16_compare_program,
                                           &compiled_remaining_f16_compare_program,
                                           &error_message),
                "expected compiled remaining F16 compare program success") ||
        !Expect(compiled_remaining_f16_compare_program[1].opcode ==
                    test_case.compiled_opcode,
                "expected compiled remaining F16 compare opcode")) {
      return 1;
    }

    WaveExecutionState compiled_remaining_f16_compare_state;
    initialize_f16_compare_state(&compiled_remaining_f16_compare_state);
    if (!Expect(interpreter.ExecuteProgram(
                    compiled_remaining_f16_compare_program,
                    &compiled_remaining_f16_compare_state, &error_message),
                "expected compiled remaining F16 compare execution success")
        ||
        !Expect(compiled_remaining_f16_compare_state.vcc_mask ==
                    (test_case.writes_exec
                         ? test_case.expected_mask
                         : (test_case.expected_mask | 0x80u)),
                "expected compiled remaining F16 compare VCC mask") ||
        !Expect(compiled_remaining_f16_compare_state.exec_mask ==
                    (test_case.writes_exec ? test_case.expected_mask : 0xfu),
                "expected compiled remaining F16 compare EXEC mask") ||
        !Expect(compiled_remaining_f16_compare_state.halted,
                "expected compiled remaining F16 compare program to halt") ||
        !Expect(compiled_remaining_f16_compare_state.pc == 2u,
                "expected compiled remaining F16 compare program advance")) {
      return 1;
    }
  }

  const std::array<std::uint32_t, 4> f16_class_cndmask_words{
      MakeSopk(0u, 124u, static_cast<std::uint16_t>(kQuietNaNF16Bits)),
      MakeVopc(125u, 124u, 98u),
      MakeVop2(1u, 60u, 314u, 59u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> f16_class_cndmask_program;
  if (!Expect(decoder.DecodeProgram(f16_class_cndmask_words,
                                    &f16_class_cndmask_program,
                                    &error_message),
              "expected F16 class cndmask program decode success") ||
      !Expect(f16_class_cndmask_program.size() == 4u,
              "expected four decoded F16 class cndmask instructions") ||
      !Expect(f16_class_cndmask_program[1].opcode == "V_CMP_CLASS_F16",
              "expected decoded V_CMP_CLASS_F16") ||
      !Expect(f16_class_cndmask_program[2].opcode == "V_CNDMASK_B32",
              "expected decoded V_CNDMASK_B32 after F16 class")) {
    return 1;
  }

  auto initialize_f16_class_cndmask_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xfu;
    state->vgprs[58][0] = 10u;
    state->vgprs[58][1] = 10u;
    state->vgprs[58][2] = 10u;
    state->vgprs[58][3] = 10u;
    state->vgprs[59][0] = 20u;
    state->vgprs[59][1] = 20u;
    state->vgprs[59][2] = 20u;
    state->vgprs[59][3] = 20u;
    state->vgprs[98][0] = 0x002u;
    state->vgprs[98][1] = 0x001u;
    state->vgprs[98][2] = 0x002u;
    state->vgprs[98][3] = 0x200u;
  };

  WaveExecutionState decoded_f16_class_cndmask_state;
  initialize_f16_class_cndmask_state(&decoded_f16_class_cndmask_state);
  if (!Expect(interpreter.ExecuteProgram(f16_class_cndmask_program,
                                         &decoded_f16_class_cndmask_state,
                                         &error_message),
              "expected decoded F16 class cndmask execution success") ||
      !Expect(ExpectF16ClassCndmaskState(decoded_f16_class_cndmask_state),
              "expected decoded F16 class cndmask state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_f16_class_cndmask_program;
  if (!Expect(interpreter.CompileProgram(f16_class_cndmask_program,
                                         &compiled_f16_class_cndmask_program,
                                         &error_message),
              "expected compiled F16 class cndmask program success") ||
      !Expect(compiled_f16_class_cndmask_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVCmpClassF16,
              "expected compiled V_CMP_CLASS_F16 opcode") ||
      !Expect(compiled_f16_class_cndmask_program[2].opcode ==
                  Gfx1201CompiledOpcode::kVCndmaskB32,
              "expected compiled V_CNDMASK_B32 after F16 class")) {
    return 1;
  }

  WaveExecutionState compiled_f16_class_cndmask_state;
  initialize_f16_class_cndmask_state(&compiled_f16_class_cndmask_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_f16_class_cndmask_program,
                                         &compiled_f16_class_cndmask_state,
                                         &error_message),
              "expected compiled F16 class cndmask execution success") ||
      !Expect(ExpectF16ClassCndmaskState(compiled_f16_class_cndmask_state),
              "expected compiled F16 class cndmask state")) {
    return 1;
  }

  const std::array<std::uint32_t, 7> f16_cmpx_class_branch_words{
      MakeSopk(0u, 114u, static_cast<std::uint16_t>(kQuietNaNF16Bits)),
      MakeVopc(253u, 114u, 99u),
      MakeSopp(38u, 2u),
      MakeSopk(0u, 119u, 111u),
      MakeSopp(32u, 1u),
      MakeSopk(0u, 119u, 222u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> f16_cmpx_class_branch_program;
  if (!Expect(decoder.DecodeProgram(f16_cmpx_class_branch_words,
                                    &f16_cmpx_class_branch_program,
                                    &error_message),
              "expected F16 CMPX class branch program decode success") ||
      !Expect(f16_cmpx_class_branch_program.size() == 7u,
              "expected seven decoded F16 CMPX class branch instructions") ||
      !Expect(f16_cmpx_class_branch_program[1].opcode == "V_CMPX_CLASS_F16",
              "expected decoded V_CMPX_CLASS_F16") ||
      !Expect(f16_cmpx_class_branch_program[2].opcode == "S_CBRANCH_EXECNZ",
              "expected decoded S_CBRANCH_EXECNZ after F16 CMPX class")) {
    return 1;
  }

  auto initialize_f16_cmpx_class_branch_state = [](WaveExecutionState* state) {
    state->exec_mask = 0xfu;
    state->vgprs[99][0] = 0x001u;
    state->vgprs[99][1] = 0x002u;
    state->vgprs[99][2] = 0x004u;
    state->vgprs[99][3] = 0x002u;
  };

  WaveExecutionState decoded_f16_cmpx_class_branch_state;
  initialize_f16_cmpx_class_branch_state(&decoded_f16_cmpx_class_branch_state);
  if (!Expect(interpreter.ExecuteProgram(f16_cmpx_class_branch_program,
                                         &decoded_f16_cmpx_class_branch_state,
                                         &error_message),
              "expected decoded F16 CMPX class branch execution success") ||
      !Expect(ExpectF16CmpxClassBranchState(
                  decoded_f16_cmpx_class_branch_state),
              "expected decoded F16 CMPX class branch state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction>
      compiled_f16_cmpx_class_branch_program;
  if (!Expect(interpreter.CompileProgram(f16_cmpx_class_branch_program,
                                         &compiled_f16_cmpx_class_branch_program,
                                         &error_message),
              "expected compiled F16 CMPX class branch program success") ||
      !Expect(compiled_f16_cmpx_class_branch_program[1].opcode ==
                  Gfx1201CompiledOpcode::kVCmpxClassF16,
              "expected compiled V_CMPX_CLASS_F16 opcode") ||
      !Expect(compiled_f16_cmpx_class_branch_program[2].opcode ==
                  Gfx1201CompiledOpcode::kSCbranchExecnz,
              "expected compiled S_CBRANCH_EXECNZ after F16 CMPX class")) {
    return 1;
  }

  WaveExecutionState compiled_f16_cmpx_class_branch_state;
  initialize_f16_cmpx_class_branch_state(&compiled_f16_cmpx_class_branch_state);
  if (!Expect(interpreter.ExecuteProgram(compiled_f16_cmpx_class_branch_program,
                                         &compiled_f16_cmpx_class_branch_state,
                                         &error_message),
              "expected compiled F16 CMPX class branch execution success") ||
      !Expect(ExpectF16CmpxClassBranchState(
                  compiled_f16_cmpx_class_branch_state),
              "expected compiled F16 CMPX class branch state")) {
    return 1;
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
