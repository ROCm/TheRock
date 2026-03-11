#ifndef MIRAGE_SIM_ISA_COMMON_DECODED_INSTRUCTION_H_
#define MIRAGE_SIM_ISA_COMMON_DECODED_INSTRUCTION_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string_view>

#include "lib/sim/isa/common/operand_metadata.h"

namespace mirage::sim::isa {

enum class OperandKind {
  kSgpr,
  kVgpr,
  kImm32,
};

struct InstructionOperand {
  OperandKind kind = OperandKind::kImm32;
  std::uint16_t index = 0;
  std::uint32_t imm32 = 0;
  OperandDescriptor descriptor{};

  static InstructionOperand Sgpr(std::uint16_t index_value,
                                 OperandDescriptor descriptor_value = {}) {
    InstructionOperand operand;
    operand.kind = OperandKind::kSgpr;
    operand.index = index_value;
    operand.descriptor = descriptor_value;
    return operand;
  }

  static InstructionOperand Vgpr(std::uint16_t index_value,
                                 OperandDescriptor descriptor_value = {}) {
    InstructionOperand operand;
    operand.kind = OperandKind::kVgpr;
    operand.index = index_value;
    operand.descriptor = descriptor_value;
    return operand;
  }

  static InstructionOperand Imm32(std::uint32_t value,
                                  OperandDescriptor descriptor_value = {}) {
    InstructionOperand operand;
    operand.kind = OperandKind::kImm32;
    operand.imm32 = value;
    operand.descriptor = descriptor_value;
    return operand;
  }

  InstructionOperand WithDescriptor(OperandDescriptor descriptor_value) const {
    InstructionOperand operand = *this;
    operand.descriptor = descriptor_value;
    return operand;
  }
};

struct DecodedInstruction {
  static constexpr std::size_t kMaxOperands = 8;

  std::string_view opcode;
  std::array<InstructionOperand, kMaxOperands> operands{};
  std::uint8_t operand_count = 0;

 private:
  template <typename... OperandTypes>
  static DecodedInstruction MakeWithOperands(std::string_view opcode_value,
                                             OperandTypes... operand_values) {
    static_assert(sizeof...(operand_values) <= kMaxOperands);
    DecodedInstruction instruction;
    instruction.opcode = opcode_value;
    std::size_t operand_index = 0;
    (void)std::initializer_list<int>{
        ((instruction.operands[operand_index++] = operand_values), 0)...};
    instruction.operand_count = static_cast<std::uint8_t>(operand_index);
    return instruction;
  }

 public:

  static DecodedInstruction Nullary(std::string_view opcode_value) {
    DecodedInstruction instruction;
    instruction.opcode = opcode_value;
    return instruction;
  }

  static DecodedInstruction OneOperand(std::string_view opcode_value,
                                       InstructionOperand operand0) {
    return MakeWithOperands(opcode_value, operand0);
  }

  static DecodedInstruction TwoOperand(std::string_view opcode_value,
                                       InstructionOperand operand0,
                                       InstructionOperand operand1) {
    return MakeWithOperands(opcode_value, operand0, operand1);
  }

  static DecodedInstruction ThreeOperand(std::string_view opcode_value,
                                         InstructionOperand operand0,
                                         InstructionOperand operand1,
                                         InstructionOperand operand2) {
    return MakeWithOperands(opcode_value, operand0, operand1, operand2);
  }

  static DecodedInstruction FourOperand(std::string_view opcode_value,
                                        InstructionOperand operand0,
                                        InstructionOperand operand1,
                                        InstructionOperand operand2,
                                        InstructionOperand operand3) {
    return MakeWithOperands(opcode_value, operand0, operand1, operand2,
                            operand3);
  }

  static DecodedInstruction FiveOperand(std::string_view opcode_value,
                                        InstructionOperand operand0,
                                        InstructionOperand operand1,
                                        InstructionOperand operand2,
                                        InstructionOperand operand3,
                                        InstructionOperand operand4) {
    return MakeWithOperands(opcode_value, operand0, operand1, operand2,
                            operand3, operand4);
  }

  static DecodedInstruction SixOperand(std::string_view opcode_value,
                                       InstructionOperand operand0,
                                       InstructionOperand operand1,
                                       InstructionOperand operand2,
                                       InstructionOperand operand3,
                                       InstructionOperand operand4,
                                       InstructionOperand operand5) {
    return MakeWithOperands(opcode_value, operand0, operand1, operand2,
                            operand3, operand4, operand5);
  }

  static DecodedInstruction SevenOperand(std::string_view opcode_value,
                                         InstructionOperand operand0,
                                         InstructionOperand operand1,
                                         InstructionOperand operand2,
                                         InstructionOperand operand3,
                                         InstructionOperand operand4,
                                         InstructionOperand operand5,
                                         InstructionOperand operand6) {
    return MakeWithOperands(opcode_value, operand0, operand1, operand2,
                            operand3, operand4, operand5, operand6);
  }

  static DecodedInstruction EightOperand(std::string_view opcode_value,
                                         InstructionOperand operand0,
                                         InstructionOperand operand1,
                                         InstructionOperand operand2,
                                         InstructionOperand operand3,
                                         InstructionOperand operand4,
                                         InstructionOperand operand5,
                                         InstructionOperand operand6,
                                         InstructionOperand operand7) {
    return MakeWithOperands(opcode_value, operand0, operand1, operand2,
                            operand3, operand4, operand5, operand6, operand7);
  }

  static DecodedInstruction Unary(std::string_view opcode_value,
                                  InstructionOperand dst,
                                  InstructionOperand src) {
    return TwoOperand(opcode_value, dst, src);
  }

  static DecodedInstruction Binary(std::string_view opcode_value,
                                   InstructionOperand dst,
                                   InstructionOperand src0,
                                   InstructionOperand src1) {
    return ThreeOperand(opcode_value, dst, src0, src1);
  }

  static DecodedInstruction Ternary(std::string_view opcode_value,
                                    InstructionOperand dst,
                                    InstructionOperand src0,
                                    InstructionOperand src1,
                                    InstructionOperand src2) {
    return FourOperand(opcode_value, dst, src0, src1, src2);
  }
};

}  // namespace mirage::sim::isa

#endif  // MIRAGE_SIM_ISA_COMMON_DECODED_INSTRUCTION_H_
