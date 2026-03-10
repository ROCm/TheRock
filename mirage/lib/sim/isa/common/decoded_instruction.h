#ifndef MIRAGE_SIM_ISA_COMMON_DECODED_INSTRUCTION_H_
#define MIRAGE_SIM_ISA_COMMON_DECODED_INSTRUCTION_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

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

  static InstructionOperand Sgpr(std::uint16_t index_value) {
    InstructionOperand operand;
    operand.kind = OperandKind::kSgpr;
    operand.index = index_value;
    return operand;
  }

  static InstructionOperand Vgpr(std::uint16_t index_value) {
    InstructionOperand operand;
    operand.kind = OperandKind::kVgpr;
    operand.index = index_value;
    return operand;
  }

  static InstructionOperand Imm32(std::uint32_t value) {
    InstructionOperand operand;
    operand.kind = OperandKind::kImm32;
    operand.imm32 = value;
    return operand;
  }
};

struct DecodedInstruction {
  static constexpr std::size_t kMaxOperands = 5;

  std::string_view opcode;
  std::array<InstructionOperand, kMaxOperands> operands{};
  std::uint8_t operand_count = 0;

  static DecodedInstruction Nullary(std::string_view opcode_value) {
    DecodedInstruction instruction;
    instruction.opcode = opcode_value;
    return instruction;
  }

  static DecodedInstruction OneOperand(std::string_view opcode_value,
                                       InstructionOperand operand0) {
    DecodedInstruction instruction;
    instruction.opcode = opcode_value;
    instruction.operands = {operand0, {}, {}, {}, {}};
    instruction.operand_count = 1;
    return instruction;
  }

  static DecodedInstruction TwoOperand(std::string_view opcode_value,
                                       InstructionOperand operand0,
                                       InstructionOperand operand1) {
    DecodedInstruction instruction;
    instruction.opcode = opcode_value;
    instruction.operands = {operand0, operand1, {}, {}, {}};
    instruction.operand_count = 2;
    return instruction;
  }

  static DecodedInstruction ThreeOperand(std::string_view opcode_value,
                                         InstructionOperand operand0,
                                         InstructionOperand operand1,
                                         InstructionOperand operand2) {
    DecodedInstruction instruction;
    instruction.opcode = opcode_value;
    instruction.operands = {operand0, operand1, operand2, {}, {}};
    instruction.operand_count = 3;
    return instruction;
  }

  static DecodedInstruction FourOperand(std::string_view opcode_value,
                                        InstructionOperand operand0,
                                        InstructionOperand operand1,
                                        InstructionOperand operand2,
                                        InstructionOperand operand3) {
    DecodedInstruction instruction;
    instruction.opcode = opcode_value;
    instruction.operands = {operand0, operand1, operand2, operand3, {}};
    instruction.operand_count = 4;
    return instruction;
  }

  static DecodedInstruction FiveOperand(std::string_view opcode_value,
                                        InstructionOperand operand0,
                                        InstructionOperand operand1,
                                        InstructionOperand operand2,
                                        InstructionOperand operand3,
                                        InstructionOperand operand4) {
    DecodedInstruction instruction;
    instruction.opcode = opcode_value;
    instruction.operands = {operand0, operand1, operand2, operand3, operand4};
    instruction.operand_count = 5;
    return instruction;
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
