#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/common/numeric_conversions.h"
#include "lib/sim/isa/common/operand_metadata.h"

namespace {

using mirage::sim::isa::DecodedInstruction;
using mirage::sim::isa::FragmentKind;
using mirage::sim::isa::InstructionOperand;
using mirage::sim::isa::MakeMatrixFragmentShape;
using mirage::sim::isa::MakePackedFragmentShape;
using mirage::sim::isa::OperandAccess;
using mirage::sim::isa::OperandDescriptor;
using mirage::sim::isa::OperandRole;
using mirage::sim::isa::OperandSlotKind;
using mirage::sim::isa::OperandValueClass;

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main() {
  OperandDescriptor descriptor;
  descriptor.role = OperandRole::kSource2;
  descriptor.slot_kind = OperandSlotKind::kSource2;
  descriptor.value_class = OperandValueClass::kPackedVector;
  descriptor.access = OperandAccess::kReadWrite;
  descriptor.fragment_shape = MakePackedFragmentShape(2, 16);
  descriptor.component_count = 2;
  descriptor.is_implicit = true;

  const DecodedInstruction instruction = DecodedInstruction::EightOperand(
      "TEST_OP", InstructionOperand::Sgpr(0),
      InstructionOperand::Sgpr(2),
      InstructionOperand::Vgpr(4),
      InstructionOperand::Vgpr(6),
      InstructionOperand::Imm32(8),
      InstructionOperand::Vgpr(10, descriptor),
      InstructionOperand::Sgpr(12), InstructionOperand::Imm32(14));

  if (!Expect(instruction.operand_count == 8,
              "expected eight-operand instruction support")) {
    return 1;
  }
  if (!Expect(instruction.operands[5].descriptor.role == OperandRole::kSource2,
              "expected operand descriptor role to survive construction")) {
    return 1;
  }
  if (!Expect(instruction.operands[5].descriptor.fragment_shape.kind ==
                  FragmentKind::kPacked &&
                  instruction.operands[5].descriptor.fragment_shape.packed_elements ==
                      2 &&
                  instruction.operands[5].descriptor.fragment_shape.element_bit_width ==
                      16,
              "expected packed fragment metadata to survive construction")) {
    return 1;
  }

  const auto matrix_shape = MakeMatrixFragmentShape(16, 16, 4, 32, 32);
  if (!Expect(matrix_shape.kind == FragmentKind::kMatrix &&
                  matrix_shape.rows == 16 &&
                  matrix_shape.columns == 16 &&
                  matrix_shape.depth == 4 &&
                  matrix_shape.element_bit_width == 32 &&
                  matrix_shape.wave_size == 32,
              "expected matrix fragment helper to preserve shape")) {
    return 1;
  }

  if (!Expect(mirage::sim::isa::FloatToHalf(1.0f) == 0x3c00u,
              "expected 1.0f to round-trip to half")) {
    return 1;
  }
  if (!Expect(std::fabs(mirage::sim::isa::HalfToFloat(0x3c00u) - 1.0f) <
                  1.0e-6f,
              "expected half 1.0 to convert back to float")) {
    return 1;
  }
  if (!Expect(mirage::sim::isa::FloatToBFloat16(1.0f) == 0x3f80u,
              "expected 1.0f to round-trip to bfloat16")) {
    return 1;
  }
  if (!Expect(std::fabs(mirage::sim::isa::BFloat16ToFloat(0x3f80u) - 1.0f) <
                  1.0e-6f,
              "expected bfloat16 1.0 to convert back to float")) {
    return 1;
  }
  if (!Expect(mirage::sim::isa::PackedHalfAdd(0x3c003c00u, 0x3c003c00u) ==
                  0x40004000u,
              "expected packed half add to add both lanes")) {
    return 1;
  }
  if (!Expect(mirage::sim::isa::PackedBFloat16Add(0x3f803f80u, 0x3f803f80u) ==
                  0x40004000u,
              "expected packed bfloat16 add to add both lanes")) {
    return 1;
  }
  if (!Expect(mirage::sim::isa::TruncateFloatToI16(
                  std::numeric_limits<float>::quiet_NaN()) == 0,
              "expected NaN truncation to match existing interpreter behavior")) {
    return 1;
  }

  return 0;
}
