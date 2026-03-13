#include <iostream>
#include <string_view>

#include "lib/sim/isa/gfx1250/stub_decoder.h"

namespace {

using mirage::sim::isa::gfx1250::DecodeMimgTensorStub;
using mirage::sim::isa::gfx1250::DecodeStubInstruction;
using mirage::sim::isa::gfx1250::DecodeVop1Stub;
using mirage::sim::isa::gfx1250::DecodeVop3SdstStub;
using mirage::sim::isa::gfx1250::DecodeVop3pStub;
using mirage::sim::isa::gfx1250::FindStubDecoderEntrypointManifest;
using mirage::sim::isa::gfx1250::FindStubDecoderRouteInfo;
using mirage::sim::isa::gfx1250::GetStubDecoderEntrypointManifests;
using mirage::sim::isa::gfx1250::GetStubDecoderRouteInstructions;
using mirage::sim::isa::gfx1250::GetStubExecutionDomainName;
using mirage::sim::isa::gfx1250::GetStubOperandLayoutName;
using mirage::sim::isa::gfx1250::GetStubOperandRoleName;
using mirage::sim::isa::gfx1250::GetStubOperandSlotKindName;
using mirage::sim::isa::gfx1250::GetStubOperandValueClassName;
using mirage::sim::isa::gfx1250::GetStubOpcodeShapeName;
using mirage::sim::isa::gfx1250::GetStubDecoderRouteInfos;
using mirage::sim::isa::gfx1250::StubDecodedInstruction;
using mirage::sim::isa::gfx1250::StubDecodeStatus;
using mirage::sim::isa::gfx1250::StubDecoderEntrypointManifest;
using mirage::sim::isa::gfx1250::StubDecoderRoute;
using mirage::sim::isa::gfx1250::StubExecutionDomain;
using mirage::sim::isa::gfx1250::StubOperandAccess;
using mirage::sim::isa::gfx1250::StubFragmentKind;
using mirage::sim::isa::gfx1250::StubOperandLayoutKind;
using mirage::sim::isa::gfx1250::StubOperandRole;
using mirage::sim::isa::gfx1250::StubOperandSlotKind;
using mirage::sim::isa::gfx1250::StubOperandValueClass;
using mirage::sim::isa::gfx1250::StubOpcodeShape;

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

bool ContainsRole(const StubDecodedInstruction& instruction,
                  StubOperandRole role,
                  std::uint32_t count,
                  bool is_output) {
  for (std::uint32_t i = 0; i < instruction.operand_roles.binding_count; ++i) {
    const auto& binding = instruction.operand_roles.bindings[i];
    if (binding.role == role && binding.count == count &&
        binding.is_output == is_output) {
      return true;
    }
  }
  return false;
}

bool ContainsSlot(const StubDecodedInstruction& instruction,
                  StubOperandSlotKind slot_kind,
                  StubOperandValueClass value_class,
                  std::uint32_t logical_operand_index,
                  std::uint32_t component_count,
                  bool is_output) {
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    const auto& binding = instruction.operand_slots.bindings[i];
    if (binding.slot_kind == slot_kind &&
        binding.value_class == value_class &&
        binding.logical_operand_index == logical_operand_index &&
        binding.component_count == component_count &&
        binding.is_output == is_output) {
      return true;
    }
  }
  return false;
}

bool ContainsSlotFragment(const StubDecodedInstruction& instruction,
                          StubOperandSlotKind slot_kind,
                          StubFragmentKind fragment_kind,
                          std::uint16_t rows,
                          std::uint16_t columns,
                          std::uint16_t depth,
                          std::uint8_t element_bit_width,
                          std::uint8_t packed_elements) {
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    const auto& binding = instruction.operand_slots.bindings[i];
    if (binding.slot_kind == slot_kind &&
        binding.fragment_shape.kind == fragment_kind &&
        binding.fragment_shape.rows == rows &&
        binding.fragment_shape.columns == columns &&
        binding.fragment_shape.depth == depth &&
        binding.fragment_shape.element_bit_width == element_bit_width &&
        binding.fragment_shape.packed_elements == packed_elements) {
      return true;
    }
  }
  return false;
}

bool ContainsSlotWaveSize(const StubDecodedInstruction& instruction,
                          StubOperandSlotKind slot_kind,
                          StubFragmentKind fragment_kind,
                          std::uint8_t wave_size) {
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    const auto& binding = instruction.operand_slots.bindings[i];
    if (binding.slot_kind == slot_kind &&
        binding.fragment_shape.kind == fragment_kind &&
        binding.fragment_shape.wave_size == wave_size) {
      return true;
    }
  }
  return false;
}

bool ContainsDescriptor(const StubDecodedInstruction& instruction,
                        StubOperandRole role,
                        StubOperandSlotKind slot_kind,
                        StubOperandValueClass value_class,
                        StubOperandAccess access,
                        std::uint8_t component_count,
                        StubFragmentKind fragment_kind,
                        std::uint8_t element_bit_width) {
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    const auto& descriptor = instruction.operand_descriptors.descriptors[i];
    if (descriptor.role == role && descriptor.slot_kind == slot_kind &&
        descriptor.value_class == value_class && descriptor.access == access &&
        descriptor.component_count == component_count &&
        descriptor.fragment_shape.kind == fragment_kind &&
        descriptor.fragment_shape.element_bit_width == element_bit_width) {
      return true;
    }
  }
  return false;
}

bool ContainsDescriptorWaveSize(const StubDecodedInstruction& instruction,
                                StubOperandRole role,
                                StubOperandSlotKind slot_kind,
                                StubFragmentKind fragment_kind,
                                std::uint8_t wave_size) {
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    const auto& descriptor = instruction.operand_descriptors.descriptors[i];
    if (descriptor.role == role && descriptor.slot_kind == slot_kind &&
        descriptor.fragment_shape.kind == fragment_kind &&
        descriptor.fragment_shape.wave_size == wave_size) {
      return true;
    }
  }
  return false;
}

bool HasMatrixSlot(const StubDecodedInstruction& instruction) {
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    if (instruction.operand_slots.bindings[i].fragment_shape.kind ==
        StubFragmentKind::kMatrix) {
      return true;
    }
  }
  return false;
}

bool AllMatrixSlotsHaveWaveSize(const StubDecodedInstruction& instruction,
                                std::uint8_t wave_size) {
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    const auto& binding = instruction.operand_slots.bindings[i];
    if (binding.fragment_shape.kind == StubFragmentKind::kMatrix &&
        binding.fragment_shape.wave_size != wave_size) {
      return false;
    }
  }
  return true;
}

bool HasMatrixDescriptor(const StubDecodedInstruction& instruction) {
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    if (instruction.operand_descriptors.descriptors[i].fragment_shape.kind ==
        StubFragmentKind::kMatrix) {
      return true;
    }
  }
  return false;
}

bool AllMatrixDescriptorsHaveWaveSize(const StubDecodedInstruction& instruction,
                                      std::uint8_t wave_size) {
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    const auto& descriptor = instruction.operand_descriptors.descriptors[i];
    if (descriptor.fragment_shape.kind == StubFragmentKind::kMatrix &&
        descriptor.fragment_shape.wave_size != wave_size) {
      return false;
    }
  }
  return true;
}

bool AllSlotWaveSizesAre(const StubDecodedInstruction& instruction,
                         std::uint8_t wave_size) {
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    if (instruction.operand_slots.bindings[i].fragment_shape.wave_size !=
        wave_size) {
      return false;
    }
  }
  return true;
}

bool AllDescriptorWaveSizesAre(const StubDecodedInstruction& instruction,
                               std::uint8_t wave_size) {
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    if (instruction.operand_descriptors.descriptors[i].fragment_shape.wave_size !=
        wave_size) {
      return false;
    }
  }
  return true;
}

bool HasDescriptorRole(const StubDecodedInstruction& instruction,
                       StubOperandRole role) {
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    if (instruction.operand_descriptors.descriptors[i].role == role) {
      return true;
    }
  }
  return false;
}

bool HasSlotKind(const StubDecodedInstruction& instruction,
                 StubOperandSlotKind slot_kind) {
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    if (instruction.operand_slots.bindings[i].slot_kind == slot_kind) {
      return true;
    }
  }
  return false;
}

bool AllSlotsExplicit(const StubDecodedInstruction& instruction) {
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    if (instruction.operand_slots.bindings[i].is_implicit) {
      return false;
    }
  }
  return true;
}

bool AllDescriptorsExplicit(const StubDecodedInstruction& instruction) {
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    if (instruction.operand_descriptors.descriptors[i].is_implicit) {
      return false;
    }
  }
  return true;
}

std::uint32_t CountDescriptorsForRole(const StubDecodedInstruction& instruction,
                                      StubOperandRole role) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    if (instruction.operand_descriptors.descriptors[i].role == role) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountSlotsOfKind(const StubDecodedInstruction& instruction,
                               StubOperandSlotKind slot_kind) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    if (instruction.operand_slots.bindings[i].slot_kind == slot_kind) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountDescriptorsWithAccess(
    const StubDecodedInstruction& instruction,
    StubOperandAccess access) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    if (instruction.operand_descriptors.descriptors[i].access == access) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountOutputSlots(const StubDecodedInstruction& instruction) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    if (instruction.operand_slots.bindings[i].is_output) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountSlotsWithFragmentKindAndWaveSize(
    const StubDecodedInstruction& instruction,
    StubFragmentKind fragment_kind,
    std::uint8_t wave_size) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    const auto& binding = instruction.operand_slots.bindings[i];
    if (binding.fragment_shape.kind == fragment_kind &&
        binding.fragment_shape.wave_size == wave_size) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountDescriptorsWithFragmentKindAndWaveSize(
    const StubDecodedInstruction& instruction,
    StubFragmentKind fragment_kind,
    std::uint8_t wave_size) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    const auto& descriptor = instruction.operand_descriptors.descriptors[i];
    if (descriptor.fragment_shape.kind == fragment_kind &&
        descriptor.fragment_shape.wave_size == wave_size) {
      ++count;
    }
  }
  return count;
}

}  // namespace

int main() {
  if (!Expect(GetStubDecoderEntrypointManifests().size() == 4,
              "expected four stub decoder entrypoint manifests")) {
    return 1;
  }
  if (!Expect(!GetStubDecoderRouteInstructions(StubDecoderRoute::kVop3p).empty(),
              "expected VOP3P routed instruction list")) {
    return 1;
  }
  if (!Expect(
          !GetStubDecoderRouteInstructions(StubDecoderRoute::kMimgTensor).empty(),
          "expected tensor routed instruction list")) {
    return 1;
  }
  if (!Expect(!GetStubDecoderRouteInstructions(StubDecoderRoute::kVop1).empty(),
              "expected VOP1 routed instruction list")) {
    return 1;
  }
  if (!Expect(
          !GetStubDecoderRouteInstructions(StubDecoderRoute::kVop3Sdst).empty(),
          "expected VOP3 SDST routed instruction list")) {
    return 1;
  }

  const StubDecodedInstruction vop3p = DecodeStubInstruction("V_PK_ADD_BF16");
  if (!Expect(vop3p.status == StubDecodeStatus::kDecodedStub,
              "expected V_PK_ADD_BF16 to decode through stub path")) {
    return 1;
  }
  if (!Expect(vop3p.route == StubDecoderRoute::kVop3p,
              "expected VOP3P route for V_PK_ADD_BF16")) {
    return 1;
  }
  if (!Expect(vop3p.entrypoint_name == "DecodeVop3pStub",
              "expected VOP3P entrypoint name")) {
    return 1;
  }
  if (!Expect(vop3p.opcode_shape == StubOpcodeShape::kVop3pPackedBinary,
              "expected packed-binary shape for V_PK_ADD_BF16")) {
    return 1;
  }
  if (!Expect(vop3p.execution_domain == StubExecutionDomain::kVectorAlu,
              "expected vector-ALU domain for V_PK_ADD_BF16")) {
    return 1;
  }
  if (!Expect(vop3p.uses_paired_operands,
              "expected packed VOP3P op to use paired operands")) {
    return 1;
  }
  if (!Expect(vop3p.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kPkAddBf16,
              "expected PK_ADD operand layout")) {
    return 1;
  }
  if (!Expect(vop3p.operand_layout.source_count == 2 &&
                  vop3p.operand_layout.destination_count == 1,
              "expected PK_ADD operand layout counts")) {
    return 1;
  }
  if (!Expect(vop3p.operand_roles.binding_count == 3,
              "expected PK_ADD operand role count")) {
    return 1;
  }
  if (!Expect(ContainsRole(vop3p, StubOperandRole::kSource0, 1, false) &&
                  ContainsRole(vop3p, StubOperandRole::kSource1, 1, false) &&
                  ContainsRole(vop3p, StubOperandRole::kDestination, 1, true),
              "expected PK_ADD operand roles")) {
    return 1;
  }
  if (!Expect(vop3p.operand_slots.binding_count == 3,
              "expected PK_ADD operand slot count")) {
    return 1;
  }
  if (!Expect(vop3p.operand_descriptors.descriptor_count == 3,
              "expected PK_ADD descriptor count")) {
    return 1;
  }
  if (!Expect(
          ContainsSlot(vop3p, StubOperandSlotKind::kDestination,
                       StubOperandValueClass::kPackedVector, 0, 2, true) &&
              ContainsSlot(vop3p, StubOperandSlotKind::kSource0,
                           StubOperandValueClass::kPackedVector, 1, 2, false) &&
              ContainsSlot(vop3p, StubOperandSlotKind::kSource1,
                           StubOperandValueClass::kPackedVector, 2, 2, false),
          "expected PK_ADD operand slots")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(vop3p, StubOperandSlotKind::kDestination,
                                   StubFragmentKind::kPacked, 1, 1, 1, 16, 2),
              "expected PK_ADD destination fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(vop3p, StubOperandRole::kDestination,
                                 StubOperandSlotKind::kDestination,
                                 StubOperandValueClass::kPackedVector,
                                 StubOperandAccess::kWrite, 2,
                                 StubFragmentKind::kPacked, 16),
              "expected PK_ADD destination descriptor")) {
    return 1;
  }

  const StubDecodedInstruction vop3p_fma = DecodeVop3pStub("V_PK_FMA_BF16");
  if (!Expect(vop3p_fma.opcode_shape == StubOpcodeShape::kVop3pPackedFma,
              "expected packed-FMA shape for V_PK_FMA_BF16")) {
    return 1;
  }
  if (!Expect(vop3p_fma.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kPkFmaBf16,
              "expected PK_FMA operand layout")) {
    return 1;
  }
  if (!Expect(vop3p_fma.operand_layout.source_count == 3 &&
                  vop3p_fma.operand_layout.destination_count == 1,
              "expected PK_FMA operand layout counts")) {
    return 1;
  }
  if (!Expect(ContainsRole(vop3p_fma, StubOperandRole::kSource2, 1, false),
              "expected PK_FMA source2 operand role")) {
    return 1;
  }
  if (!Expect(ContainsSlot(vop3p_fma, StubOperandSlotKind::kSource2,
                           StubOperandValueClass::kPackedVector, 3, 2, false),
              "expected PK_FMA source2 operand slot")) {
    return 1;
  }

  const StubDecodedInstruction pk_mul = DecodeVop3pStub("V_PK_MUL_BF16");
  if (!Expect(pk_mul.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kPkMulBf16,
              "expected PK_MUL operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlot(pk_mul, StubOperandSlotKind::kDestination,
                           StubOperandValueClass::kPackedVector, 0, 2, true),
              "expected PK_MUL destination slot")) {
    return 1;
  }

  const StubDecodedInstruction pk_max_num =
      DecodeVop3pStub("V_PK_MAX_NUM_BF16");
  if (!Expect(pk_max_num.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kPkMaxNumBf16,
              "expected PK_MAX_NUM operand layout")) {
    return 1;
  }
  if (!Expect(ContainsRole(pk_max_num, StubOperandRole::kSource1, 1, false),
              "expected PK_MAX_NUM source1 role")) {
    return 1;
  }

  const StubDecodedInstruction pk_min_num =
      DecodeVop3pStub("V_PK_MIN_NUM_BF16");
  if (!Expect(pk_min_num.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kPkMinNumBf16,
              "expected PK_MIN_NUM operand layout")) {
    return 1;
  }

  const StubDecodedInstruction wmma =
      DecodeVop3pStub("V_WMMA_F32_16X16X4_F32_w32");
  if (!Expect(wmma.opcode_shape == StubOpcodeShape::kWmmaCore,
              "expected WMMA core shape for WMMA seed")) {
    return 1;
  }
  if (!Expect(wmma.execution_domain == StubExecutionDomain::kMatrix,
              "expected matrix domain for WMMA seed")) {
    return 1;
  }
  if (!Expect(wmma.uses_accumulator,
              "expected WMMA seed to consume accumulator path")) {
    return 1;
  }
  if (!Expect(wmma.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaF32_16x16x4_F32W32,
              "expected WMMA core operand layout")) {
    return 1;
  }
  if (!Expect(wmma.operand_layout.source_count == 2 &&
                  wmma.operand_layout.destination_count == 1 &&
                  wmma.operand_layout.accumulator_source_count == 1,
              "expected WMMA operand layout counts")) {
    return 1;
  }
  if (!Expect(ContainsRole(wmma, StubOperandRole::kAccumulator, 1, false) &&
                  ContainsRole(wmma, StubOperandRole::kDestination, 1, true),
              "expected WMMA operand roles")) {
    return 1;
  }
  if (!Expect(
          ContainsSlot(wmma, StubOperandSlotKind::kAccumulatorSource,
                       StubOperandValueClass::kAccumulatorFragment, 3, 1,
                       false),
          "expected WMMA accumulator slot")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma,
                                   StubOperandSlotKind::kAccumulatorSource,
                                   StubFragmentKind::kMatrix, 16, 16, 4, 32,
                                   0),
              "expected WMMA accumulator fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(wmma, StubOperandRole::kAccumulator,
                                 StubOperandSlotKind::kAccumulatorSource,
                                 StubOperandValueClass::kAccumulatorFragment,
                                 StubOperandAccess::kRead, 1,
                                 StubFragmentKind::kMatrix, 32),
              "expected WMMA accumulator descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma, StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 4, 32,
                                   0),
              "expected WMMA source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(wmma, StubOperandRole::kDestination,
                                 StubOperandSlotKind::kDestination,
                                 StubOperandValueClass::kMatrixFragment,
                                 StubOperandAccess::kWrite, 1,
                                 StubFragmentKind::kMatrix, 32),
              "expected WMMA destination descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotWaveSize(wmma, StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 32) &&
                  ContainsSlotWaveSize(wmma,
                                       StubOperandSlotKind::kAccumulatorSource,
                                       StubFragmentKind::kMatrix, 32) &&
                  ContainsDescriptorWaveSize(
                      wmma, StubOperandRole::kDestination,
                      StubOperandSlotKind::kDestination,
                      StubFragmentKind::kMatrix, 32),
              "expected WMMA core wave32 fragment semantics")) {
    return 1;
  }

  const StubDecodedInstruction wmma128_fp8 =
      DecodeVop3pStub("V_WMMA_F32_16X16X128_FP8_FP8_w32");
  if (!Expect(wmma128_fp8.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaF32_16x16x128_Fp8Fp8W32,
              "expected WMMA 128 FP8 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma128_fp8, StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected WMMA 128 FP8 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma128_fp8, StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected WMMA 128 FP8 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma128_fp8,
                                   StubOperandSlotKind::kDestination,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 32,
                                   0),
              "expected WMMA 128 FP8 destination fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(wmma128_fp8, StubOperandRole::kAccumulator,
                                 StubOperandSlotKind::kAccumulatorSource,
                                 StubOperandValueClass::kAccumulatorFragment,
                                 StubOperandAccess::kRead, 1,
                                 StubFragmentKind::kMatrix, 32),
              "expected WMMA 128 FP8 accumulator descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotWaveSize(wmma128_fp8, StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 32) &&
                  ContainsDescriptorWaveSize(
                      wmma128_fp8, StubOperandRole::kAccumulator,
                      StubOperandSlotKind::kAccumulatorSource,
                      StubFragmentKind::kMatrix, 32),
              "expected WMMA 128 FP8 wave32 fragment semantics")) {
    return 1;
  }

  const StubDecodedInstruction wmma128_f16 =
      DecodeVop3pStub("V_WMMA_F16_16X16X128_FP8_FP8_w32");
  if (!Expect(wmma128_f16.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaF16_16x16x128_Fp8Fp8W32,
              "expected WMMA F16 128 FP8 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma128_f16,
                                   StubOperandSlotKind::kDestination,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 16,
                                   0),
              "expected WMMA F16 128 destination fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma128_f16,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected WMMA F16 128 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(wmma128_f16, StubOperandRole::kAccumulator,
                                 StubOperandSlotKind::kAccumulatorSource,
                                 StubOperandValueClass::kAccumulatorFragment,
                                 StubOperandAccess::kRead, 1,
                                 StubFragmentKind::kMatrix, 16),
              "expected WMMA F16 128 accumulator descriptor")) {
    return 1;
  }

  const StubDecodedInstruction wmma64_fp8 =
      DecodeVop3pStub("V_WMMA_F32_16X16X64_FP8_FP8_w32");
  if (!Expect(wmma64_fp8.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaF32_16x16x64_Fp8Fp8W32,
              "expected WMMA 64 FP8 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma64_fp8, StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 64, 8,
                                   0),
              "expected WMMA 64 FP8 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma64_fp8, StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 64, 8,
                                   0),
              "expected WMMA 64 FP8 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(wmma64_fp8, StubOperandRole::kAccumulator,
                                 StubOperandSlotKind::kAccumulatorSource,
                                 StubOperandValueClass::kAccumulatorFragment,
                                 StubOperandAccess::kRead, 1,
                                 StubFragmentKind::kMatrix, 32),
              "expected WMMA 64 FP8 accumulator descriptor")) {
    return 1;
  }

  const StubDecodedInstruction wmma_scale =
      DecodeVop3pStub("V_WMMA_SCALE_F32_16X16X128_F8F6F4");
  if (!Expect(wmma_scale.opcode_shape == StubOpcodeShape::kWmmaScale,
              "expected WMMA scale shape")) {
    return 1;
  }
  if (!Expect(wmma_scale.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaScaleF32_16x16x128_F8F6F4,
              "expected WMMA scale operand layout")) {
    return 1;
  }
  if (!Expect(ContainsRole(wmma_scale, StubOperandRole::kScale, 1, false),
              "expected WMMA scale role")) {
    return 1;
  }
  if (!Expect(ContainsSlot(wmma_scale, StubOperandSlotKind::kScaleSource,
                           StubOperandValueClass::kScalarRegister, 4, 1, false),
              "expected WMMA scale slot")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(wmma_scale, StubOperandRole::kScale,
                                 StubOperandSlotKind::kScaleSource,
                                 StubOperandValueClass::kScalarRegister,
                                 StubOperandAccess::kRead, 1,
                                 StubFragmentKind::kScalar, 32),
              "expected WMMA scale descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_scale, StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected WMMA scale source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(wmma_scale, StubOperandRole::kAccumulator,
                                 StubOperandSlotKind::kAccumulatorSource,
                                 StubOperandValueClass::kAccumulatorFragment,
                                 StubOperandAccess::kRead, 1,
                                 StubFragmentKind::kMatrix, 32),
              "expected WMMA scale accumulator descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotWaveSize(wmma_scale, StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 32) &&
                  ContainsDescriptorWaveSize(
                      wmma_scale, StubOperandRole::kAccumulator,
                      StubOperandSlotKind::kAccumulatorSource,
                      StubFragmentKind::kMatrix, 32),
              "expected WMMA scale wave32 fragment semantics")) {
    return 1;
  }

  const StubDecodedInstruction wmma_scale16 =
      DecodeVop3pStub("V_WMMA_SCALE16_F32_16X16X128_F8F6F4");
  if (!Expect(wmma_scale16.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaScale16F32_16x16x128_F8F6F4,
              "expected WMMA scale16 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(wmma_scale16, StubOperandRole::kScale,
                                 StubOperandSlotKind::kScaleSource,
                                 StubOperandValueClass::kScalarRegister,
                                 StubOperandAccess::kRead, 1,
                                 StubFragmentKind::kScalar, 32),
              "expected WMMA scale16 descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_scale16,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected WMMA scale16 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(wmma_scale16, StubOperandRole::kAccumulator,
                                 StubOperandSlotKind::kAccumulatorSource,
                                 StubOperandValueClass::kAccumulatorFragment,
                                 StubOperandAccess::kRead, 1,
                                 StubFragmentKind::kMatrix, 32),
              "expected WMMA scale16 accumulator descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotWaveSize(wmma_scale16, StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 32) &&
                  ContainsSlotWaveSize(wmma_scale16,
                                       StubOperandSlotKind::kSource1,
                                       StubFragmentKind::kMatrix, 32),
              "expected WMMA scale16 wave32 fragment semantics")) {
    return 1;
  }

  const StubDecodedInstruction paired_scale =
      DecodeVop3pStub("V_WMMA_LD_SCALE_PAIRED_B32");
  if (!Expect(
          paired_scale.opcode_shape == StubOpcodeShape::kWmmaScalePairedLoad,
          "expected paired WMMA scale-load shape")) {
    return 1;
  }
  if (!Expect(paired_scale.uses_scale_path,
              "expected paired WMMA scale load to use scale path")) {
    return 1;
  }
  if (!Expect(paired_scale.uses_paired_operands,
              "expected paired WMMA scale load to use paired operands")) {
    return 1;
  }
  if (!Expect(paired_scale.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaLdScalePairedB32,
              "expected paired WMMA scale-load operand layout")) {
    return 1;
  }
  if (!Expect(paired_scale.operand_layout.has_scale_operand &&
                  paired_scale.operand_layout.has_paired_scale_operand,
              "expected paired WMMA scale-load operand flags")) {
    return 1;
  }
  if (!Expect(
          ContainsRole(paired_scale, StubOperandRole::kScale, 1, false) &&
              ContainsRole(paired_scale, StubOperandRole::kPairedScale, 1, false),
          "expected paired WMMA scale-load roles")) {
    return 1;
  }
  if (!Expect(
          ContainsSlot(paired_scale, StubOperandSlotKind::kScaleSource,
                       StubOperandValueClass::kScalarRegister, 2, 1, false) &&
              ContainsSlot(paired_scale,
                           StubOperandSlotKind::kPairedScaleSource,
                           StubOperandValueClass::kScalarRegister, 3, 1,
                           false),
          "expected paired WMMA scale-load slots")) {
    return 1;
  }
  if (!Expect(
          ContainsDescriptor(paired_scale, StubOperandRole::kScale,
                             StubOperandSlotKind::kScaleSource,
                             StubOperandValueClass::kScalarRegister,
                             StubOperandAccess::kRead, 1,
                             StubFragmentKind::kScalar, 32) &&
              ContainsDescriptor(paired_scale, StubOperandRole::kPairedScale,
                                 StubOperandSlotKind::kPairedScaleSource,
                                 StubOperandValueClass::kScalarRegister,
                                 StubOperandAccess::kRead, 1,
                                 StubFragmentKind::kScalar, 32),
          "expected paired WMMA scale-load descriptors")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(paired_scale, StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kVector, 1, 1, 1, 32, 1),
              "expected paired WMMA scale-load source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(paired_scale, StubOperandRole::kSource0,
                                 StubOperandSlotKind::kSource0,
                                 StubOperandValueClass::kVectorRegister,
                                 StubOperandAccess::kRead, 1,
                                 StubFragmentKind::kVector, 32),
              "expected paired WMMA scale-load source descriptor")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(paired_scale, StubOperandRole::kDestination,
                                 StubOperandSlotKind::kDestination,
                                 StubOperandValueClass::kVectorRegister,
                                 StubOperandAccess::kWrite, 1,
                                 StubFragmentKind::kVector, 32),
              "expected paired WMMA scale-load destination descriptor")) {
    return 1;
  }

  const StubDecodedInstruction paired_scale16 =
      DecodeVop3pStub("V_WMMA_LD_SCALE16_PAIRED_B64");
  if (!Expect(paired_scale16.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaLdScale16PairedB64,
              "expected paired WMMA scale16 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(paired_scale16, StubOperandRole::kScale,
                                 StubOperandSlotKind::kScaleSource,
                                 StubOperandValueClass::kScalarRegister,
                                 StubOperandAccess::kRead, 1,
                                 StubFragmentKind::kScalar, 32),
              "expected paired WMMA scale16 scale descriptor")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(paired_scale16, StubOperandRole::kPairedScale,
                                 StubOperandSlotKind::kPairedScaleSource,
                                 StubOperandValueClass::kScalarRegister,
                                 StubOperandAccess::kRead, 1,
                                 StubFragmentKind::kScalar, 32),
              "expected paired WMMA scale16 paired-scale descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(
                  paired_scale16, StubOperandSlotKind::kSource0,
                  StubFragmentKind::kVector, 1, 1, 1, 64, 1),
              "expected paired WMMA scale16 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(
                  paired_scale16, StubOperandSlotKind::kDestination,
                  StubFragmentKind::kVector, 1, 1, 1, 64, 1),
              "expected paired WMMA scale16 destination fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(
                  paired_scale16, StubOperandSlotKind::kScaleSource,
                  StubFragmentKind::kScalar, 1, 1, 1, 32, 1),
              "expected paired WMMA scale16 scale fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(
                  paired_scale16, StubOperandSlotKind::kPairedScaleSource,
                  StubFragmentKind::kScalar, 1, 1, 1, 32, 1),
              "expected paired WMMA scale16 paired-scale fragment shape")) {
    return 1;
  }

  const StubDecodedInstruction swmmac =
      DecodeVop3pStub("V_SWMMAC_F32_16X16X128_FP8_FP8_w32");
  if (!Expect(swmmac.opcode_shape == StubOpcodeShape::kSwmmacCore,
              "expected SWMMAC shape")) {
    return 1;
  }
  if (!Expect(swmmac.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kSwmmacF32_16x16x128_Fp8Fp8W32,
              "expected SWMMAC operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlot(swmmac, StubOperandSlotKind::kAccumulatorSource,
                           StubOperandValueClass::kAccumulatorFragment, 3, 1,
                           false),
              "expected SWMMAC accumulator slot")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac, StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected SWMMAC source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac, StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected SWMMAC source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(swmmac, StubOperandRole::kAccumulator,
                                 StubOperandSlotKind::kAccumulatorSource,
                                 StubOperandValueClass::kAccumulatorFragment,
                                 StubOperandAccess::kRead, 1,
                                 StubFragmentKind::kMatrix, 32),
              "expected SWMMAC accumulator descriptor")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(swmmac, StubOperandRole::kDestination,
                                 StubOperandSlotKind::kDestination,
                                 StubOperandValueClass::kMatrixFragment,
                                 StubOperandAccess::kWrite, 1,
                                 StubFragmentKind::kMatrix, 32),
              "expected SWMMAC destination descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotWaveSize(swmmac, StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 32) &&
                  ContainsDescriptorWaveSize(
                      swmmac, StubOperandRole::kDestination,
                      StubOperandSlotKind::kDestination,
                      StubFragmentKind::kMatrix, 32),
              "expected SWMMAC wave32 fragment semantics")) {
    return 1;
  }

  const StubDecodedInstruction swmmac_f16 =
      DecodeVop3pStub("V_SWMMAC_F16_16X16X128_FP8_FP8_w32");
  if (!Expect(swmmac_f16.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kSwmmacF16_16x16x128_Fp8Fp8W32,
              "expected SWMMAC F16 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_f16,
                                   StubOperandSlotKind::kDestination,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 16,
                                   0),
              "expected SWMMAC F16 destination fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(swmmac_f16, StubOperandRole::kAccumulator,
                                 StubOperandSlotKind::kAccumulatorSource,
                                 StubOperandValueClass::kAccumulatorFragment,
                                 StubOperandAccess::kRead, 1,
                                 StubFragmentKind::kMatrix, 16),
              "expected SWMMAC F16 accumulator descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_f16, StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected SWMMAC F16 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(swmmac_f16, StubOperandRole::kDestination,
                                 StubOperandSlotKind::kDestination,
                                 StubOperandValueClass::kMatrixFragment,
                                 StubOperandAccess::kWrite, 1,
                                 StubFragmentKind::kMatrix, 16),
              "expected SWMMAC F16 destination descriptor")) {
    return 1;
  }

  const StubDecodedInstruction wmma_bf16f32_generic =
      DecodeVop3pStub("V_WMMA_BF16F32_16X16X32_BF16_w32");
  if (!Expect(wmma_bf16f32_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA core operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_bf16f32_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 32, 16,
                                   0),
              "expected generic WMMA BF16F32 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_bf16f32_generic,
                                   StubOperandSlotKind::kDestination,
                                   StubFragmentKind::kMatrix, 16, 16, 32, 32,
                                   0),
              "expected generic WMMA BF16F32 destination fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  wmma_bf16f32_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 32),
              "expected generic WMMA BF16F32 destination descriptor")) {
    return 1;
  }

  const StubDecodedInstruction wmma_i32_generic =
      DecodeVop3pStub("V_WMMA_I32_16X16X64_IU8_w32");
  if (!Expect(wmma_i32_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA I32 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_i32_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 64, 8, 0),
              "expected generic WMMA I32 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_i32_generic,
                                   StubOperandSlotKind::kDestination,
                                   StubFragmentKind::kMatrix, 16, 16, 64, 32,
                                   0),
              "expected generic WMMA I32 destination fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_i32_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 64, 8,
                                   0),
              "expected generic WMMA I32 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  wmma_i32_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 32),
              "expected generic WMMA I32 destination descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotWaveSize(wmma_i32_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 32) &&
                  ContainsDescriptorWaveSize(
                      wmma_i32_generic, StubOperandRole::kDestination,
                      StubOperandSlotKind::kDestination,
                      StubFragmentKind::kMatrix, 32),
              "expected generic WMMA I32 wave32 fragment semantics")) {
    return 1;
  }

  const StubDecodedInstruction swmmac_bf16f32_generic =
      DecodeVop3pStub("V_SWMMAC_BF16F32_16X16X64_BF16_w32");
  if (!Expect(swmmac_bf16f32_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kSwmmacCoreGeneric,
              "expected generic SWMMAC BF16F32 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_bf16f32_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 64, 16,
                                   0),
              "expected generic SWMMAC BF16F32 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_bf16f32_generic,
                                   StubOperandSlotKind::kDestination,
                                   StubFragmentKind::kMatrix, 16, 16, 64, 32,
                                   0),
              "expected generic SWMMAC BF16F32 destination fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotWaveSize(swmmac_bf16f32_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 32),
              "expected generic SWMMAC BF16F32 wave32 fragment semantics")) {
    return 1;
  }

  const StubDecodedInstruction swmmac_i32_generic =
      DecodeVop3pStub("V_SWMMAC_I32_16X16X128_IU8_w32");
  if (!Expect(swmmac_i32_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kSwmmacCoreGeneric,
              "expected generic SWMMAC I32 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_i32_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic SWMMAC I32 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  swmmac_i32_generic, StubOperandRole::kAccumulator,
                  StubOperandSlotKind::kAccumulatorSource,
                  StubOperandValueClass::kAccumulatorFragment,
                  StubOperandAccess::kRead, 1, StubFragmentKind::kMatrix, 32),
              "expected generic SWMMAC I32 accumulator descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_i32_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic SWMMAC I32 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  swmmac_i32_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 32),
              "expected generic SWMMAC I32 destination descriptor")) {
    return 1;
  }
  if (!Expect(ContainsDescriptorWaveSize(
                  swmmac_i32_generic, StubOperandRole::kAccumulator,
                  StubOperandSlotKind::kAccumulatorSource,
                  StubFragmentKind::kMatrix, 32),
              "expected generic SWMMAC I32 wave32 accumulator semantics")) {
    return 1;
  }

  const StubDecodedInstruction wmma_scale_f4_generic =
      DecodeVop3pStub("V_WMMA_SCALE_F32_32X16X128_F4_w32");
  if (!Expect(wmma_scale_f4_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaScaleGeneric,
              "expected generic WMMA scale operand layout")) {
    return 1;
  }
  if (!Expect(ContainsRole(wmma_scale_f4_generic, StubOperandRole::kScale, 1,
                           false),
              "expected generic WMMA scale role")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_scale_f4_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 32, 16, 128, 4,
                                   0),
              "expected generic WMMA scale source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_scale_f4_generic,
                                   StubOperandSlotKind::kDestination,
                                   StubFragmentKind::kMatrix, 32, 16, 128, 32,
                                   0),
              "expected generic WMMA scale destination fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_scale_f4_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 32, 16, 128, 4,
                                   0),
              "expected generic WMMA scale source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotWaveSize(wmma_scale_f4_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 32) &&
                  ContainsDescriptorWaveSize(
                      wmma_scale_f4_generic, StubOperandRole::kDestination,
                      StubOperandSlotKind::kDestination,
                      StubFragmentKind::kMatrix, 32),
              "expected generic WMMA scale F4 wave32 fragment semantics")) {
    return 1;
  }

  const StubDecodedInstruction wmma_scale16_f4_generic =
      DecodeVop3pStub("V_WMMA_SCALE16_F32_32X16X128_F4_w32");
  if (!Expect(wmma_scale16_f4_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaScaleGeneric,
              "expected generic WMMA scale16 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  wmma_scale16_f4_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 32),
              "expected generic WMMA scale16 destination descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_scale16_f4_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 32, 16, 128, 4,
                                   0),
              "expected generic WMMA scale16 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  wmma_scale16_f4_generic, StubOperandRole::kAccumulator,
                  StubOperandSlotKind::kAccumulatorSource,
                  StubOperandValueClass::kAccumulatorFragment,
                  StubOperandAccess::kRead, 1, StubFragmentKind::kMatrix, 32),
              "expected generic WMMA scale16 accumulator descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotWaveSize(wmma_scale16_f4_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 32) &&
                  ContainsDescriptorWaveSize(
                      wmma_scale16_f4_generic, StubOperandRole::kAccumulator,
                      StubOperandSlotKind::kAccumulatorSource,
                      StubFragmentKind::kMatrix, 32),
              "expected generic WMMA scale16 F4 wave32 fragment semantics")) {
    return 1;
  }

  const StubDecodedInstruction wmma_core_f8f6f4_generic =
      DecodeVop3pStub("V_WMMA_F32_16X16X128_F8F6F4");
  if (!Expect(wmma_core_f8f6f4_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA F8F6F4 core operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_core_f8f6f4_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic WMMA F8F6F4 core source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  wmma_core_f8f6f4_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 32),
              "expected generic WMMA F8F6F4 core destination descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotWaveSize(wmma_core_f8f6f4_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 32) &&
                  ContainsDescriptorWaveSize(
                      wmma_core_f8f6f4_generic, StubOperandRole::kDestination,
                      StubOperandSlotKind::kDestination,
                      StubFragmentKind::kMatrix, 32),
              "expected suffix-less WMMA F8F6F4 wave32 fragment semantics")) {
    return 1;
  }

  for (std::string_view instruction_name :
       GetStubDecoderRouteInstructions(StubDecoderRoute::kVop3p)) {
    if (instruction_name == "V_WMMA_LD_SCALE_PAIRED_B32" ||
        instruction_name == "V_WMMA_LD_SCALE16_PAIRED_B64") {
      continue;
    }
    const bool is_matrix_route =
        instruction_name.rfind("V_WMMA_", 0) == 0 ||
        instruction_name.rfind("V_SWMMAC_", 0) == 0;
    if (!is_matrix_route) {
      continue;
    }

    const StubDecodedInstruction decoded = DecodeVop3pStub(instruction_name);
    if (!Expect(decoded.status == StubDecodeStatus::kDecodedStub,
                "expected routed WMMA/SWMMAC seed to decode")) {
      return 1;
    }
    if (!Expect(HasMatrixSlot(decoded) && HasMatrixDescriptor(decoded),
                "expected routed WMMA/SWMMAC seed to materialize matrix metadata")) {
      return 1;
    }
    if (!Expect(AllSlotsExplicit(decoded) &&
                    AllDescriptorsExplicit(decoded) &&
                    ContainsSlot(decoded, StubOperandSlotKind::kDestination,
                                 StubOperandValueClass::kMatrixFragment, 0, 1,
                                 true) &&
                    ContainsSlot(decoded, StubOperandSlotKind::kSource0,
                                 StubOperandValueClass::kMatrixFragment, 1, 1,
                                 false) &&
                    ContainsSlot(decoded, StubOperandSlotKind::kSource1,
                                 StubOperandValueClass::kMatrixFragment, 2, 1,
                                 false) &&
                    ContainsSlot(decoded,
                                 StubOperandSlotKind::kAccumulatorSource,
                                 StubOperandValueClass::kAccumulatorFragment, 3,
                                 1, false) &&
                    ContainsSlotWaveSize(decoded,
                                         StubOperandSlotKind::kDestination,
                                         StubFragmentKind::kMatrix, 32) &&
                    ContainsSlotWaveSize(decoded, StubOperandSlotKind::kSource0,
                                         StubFragmentKind::kMatrix, 32) &&
                    ContainsSlotWaveSize(decoded, StubOperandSlotKind::kSource1,
                                         StubFragmentKind::kMatrix, 32) &&
                    ContainsSlotWaveSize(
                        decoded, StubOperandSlotKind::kAccumulatorSource,
                        StubFragmentKind::kMatrix, 32) &&
                    ContainsDescriptorWaveSize(decoded,
                                               StubOperandRole::kDestination,
                                               StubOperandSlotKind::kDestination,
                                               StubFragmentKind::kMatrix, 32) &&
                    ContainsDescriptorWaveSize(decoded,
                                               StubOperandRole::kSource0,
                                               StubOperandSlotKind::kSource0,
                                               StubFragmentKind::kMatrix, 32) &&
                    ContainsDescriptorWaveSize(decoded,
                                               StubOperandRole::kSource1,
                                               StubOperandSlotKind::kSource1,
                                               StubFragmentKind::kMatrix, 32) &&
                    ContainsDescriptorWaveSize(
                        decoded, StubOperandRole::kAccumulator,
                        StubOperandSlotKind::kAccumulatorSource,
                        StubFragmentKind::kMatrix, 32) &&
                    CountSlotsWithFragmentKindAndWaveSize(
                        decoded, StubFragmentKind::kMatrix, 32) ==
                        (instruction_name.rfind("V_WMMA_SCALE", 0) == 0 &&
                                 instruction_name.rfind("V_WMMA_LD_SCALE", 0) != 0
                             ? 4u
                             : 4u) &&
                    CountDescriptorsWithFragmentKindAndWaveSize(
                        decoded, StubFragmentKind::kMatrix, 32) ==
                        (instruction_name.rfind("V_WMMA_SCALE", 0) == 0 &&
                                 instruction_name.rfind("V_WMMA_LD_SCALE", 0) != 0
                             ? 4u
                             : 4u) &&
                    AllMatrixSlotsHaveWaveSize(decoded, 32) &&
                    AllMatrixDescriptorsHaveWaveSize(decoded, 32),
                "expected routed WMMA/SWMMAC matrix fragments to stay wave32")) {
      return 1;
    }
    if (instruction_name.rfind("V_WMMA_SCALE", 0) == 0 &&
        instruction_name.rfind("V_WMMA_LD_SCALE", 0) != 0) {
      if (!Expect(decoded.uses_scale_path &&
                      decoded.operand_slots.binding_count == 5 &&
                      decoded.operand_descriptors.descriptor_count == 5 &&
                      CountSlotsOfKind(decoded, StubOperandSlotKind::kDestination) == 1 &&
                      CountSlotsOfKind(decoded, StubOperandSlotKind::kSource0) == 1 &&
                      CountSlotsOfKind(decoded, StubOperandSlotKind::kSource1) == 1 &&
                      CountSlotsOfKind(decoded, StubOperandSlotKind::kAccumulatorSource) == 1 &&
                      CountSlotsOfKind(decoded, StubOperandSlotKind::kScaleSource) == 1 &&
                      CountDescriptorsForRole(decoded, StubOperandRole::kDestination) == 1 &&
                      CountDescriptorsForRole(decoded, StubOperandRole::kSource0) == 1 &&
                      CountDescriptorsForRole(decoded, StubOperandRole::kSource1) == 1 &&
                      CountDescriptorsForRole(decoded, StubOperandRole::kAccumulator) == 1 &&
                      CountDescriptorsForRole(decoded, StubOperandRole::kScale) == 1 &&
                      CountSlotsWithFragmentKindAndWaveSize(
                          decoded, StubFragmentKind::kMatrix, 32) == 4 &&
                      CountDescriptorsWithFragmentKindAndWaveSize(
                          decoded, StubFragmentKind::kMatrix, 32) == 4 &&
                      CountSlotsWithFragmentKindAndWaveSize(
                          decoded, StubFragmentKind::kScalar, 0) == 1 &&
                      CountDescriptorsWithFragmentKindAndWaveSize(
                          decoded, StubFragmentKind::kScalar, 0) == 1 &&
                      ContainsSlot(decoded, StubOperandSlotKind::kScaleSource,
                                   StubOperandValueClass::kScalarRegister, 4, 1,
                                   false) &&
                      ContainsSlotFragment(decoded,
                                           StubOperandSlotKind::kScaleSource,
                                           StubFragmentKind::kScalar, 1, 1, 1,
                                           32, 1) &&
                      ContainsSlotWaveSize(decoded,
                                           StubOperandSlotKind::kScaleSource,
                                           StubFragmentKind::kScalar, 0) &&
                      ContainsDescriptor(decoded, StubOperandRole::kScale,
                                         StubOperandSlotKind::kScaleSource,
                                         StubOperandValueClass::kScalarRegister,
                                         StubOperandAccess::kRead, 1,
                                         StubFragmentKind::kScalar, 32) &&
                      ContainsDescriptorWaveSize(decoded, StubOperandRole::kScale,
                                                 StubOperandSlotKind::kScaleSource,
                                                 StubFragmentKind::kScalar, 0),
                  "expected routed WMMA scale seed to keep scalar scale metadata")) {
        return 1;
      }
    } else {
      if (!Expect(decoded.operand_slots.binding_count == 4 &&
                      decoded.operand_descriptors.descriptor_count == 4 &&
                      !decoded.uses_scale_path &&
                      CountSlotsOfKind(decoded, StubOperandSlotKind::kDestination) == 1 &&
                      CountSlotsOfKind(decoded, StubOperandSlotKind::kSource0) == 1 &&
                      CountSlotsOfKind(decoded, StubOperandSlotKind::kSource1) == 1 &&
                      CountSlotsOfKind(decoded, StubOperandSlotKind::kAccumulatorSource) == 1 &&
                      CountSlotsOfKind(decoded, StubOperandSlotKind::kScaleSource) == 0 &&
                      CountDescriptorsForRole(decoded, StubOperandRole::kDestination) == 1 &&
                      CountDescriptorsForRole(decoded, StubOperandRole::kSource0) == 1 &&
                      CountDescriptorsForRole(decoded, StubOperandRole::kSource1) == 1 &&
                      CountDescriptorsForRole(decoded, StubOperandRole::kAccumulator) == 1 &&
                      CountDescriptorsForRole(decoded, StubOperandRole::kScale) == 0 &&
                      CountSlotsWithFragmentKindAndWaveSize(
                          decoded, StubFragmentKind::kMatrix, 32) == 4 &&
                      CountDescriptorsWithFragmentKindAndWaveSize(
                          decoded, StubFragmentKind::kMatrix, 32) == 4 &&
                      CountSlotsWithFragmentKindAndWaveSize(
                          decoded, StubFragmentKind::kScalar, 0) == 0 &&
                      CountDescriptorsWithFragmentKindAndWaveSize(
                          decoded, StubFragmentKind::kScalar, 0) == 0,
                  "expected routed WMMA/SWMMAC core seed to keep exact matrix composition")) {
        return 1;
      }
    }
  }

  const StubDecodedInstruction wmma_bf16_generic =
      DecodeVop3pStub("V_WMMA_BF16_16X16X32_BF16_w32");
  if (!Expect(wmma_bf16_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA BF16 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_bf16_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 32, 16,
                                   0),
              "expected generic WMMA BF16 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_bf16_generic,
                                   StubOperandSlotKind::kDestination,
                                   StubFragmentKind::kMatrix, 16, 16, 32, 16,
                                   0),
              "expected generic WMMA BF16 destination fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_bf16_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 32, 16,
                                   0),
              "expected generic WMMA BF16 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  wmma_bf16_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 16),
              "expected generic WMMA BF16 destination descriptor")) {
    return 1;
  }

  const StubDecodedInstruction wmma_f16_f16_generic =
      DecodeVop3pStub("V_WMMA_F16_16X16X32_F16_w32");
  if (!Expect(wmma_f16_f16_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA F16/F16 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f16_f16_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 32, 16,
                                   0),
              "expected generic WMMA F16/F16 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  wmma_f16_f16_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 16),
              "expected generic WMMA F16/F16 destination descriptor")) {
    return 1;
  }

  const StubDecodedInstruction wmma_f32_bf16_generic =
      DecodeVop3pStub("V_WMMA_F32_16X16X32_BF16_w32");
  if (!Expect(wmma_f32_bf16_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA F32/BF16 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f32_bf16_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 32, 16,
                                   0),
              "expected generic WMMA F32/BF16 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  wmma_f32_bf16_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 32),
              "expected generic WMMA F32/BF16 destination descriptor")) {
    return 1;
  }

  const StubDecodedInstruction wmma_f16_bf8_fp8_generic =
      DecodeVop3pStub("V_WMMA_F16_16X16X64_BF8_FP8_w32");
  if (!Expect(wmma_f16_bf8_fp8_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA F16 BF8/FP8 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f16_bf8_fp8_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 64, 8,
                                   0),
              "expected generic WMMA F16 BF8/FP8 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  wmma_f16_bf8_fp8_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 16),
              "expected generic WMMA F16 BF8/FP8 destination descriptor")) {
    return 1;
  }

  const StubDecodedInstruction wmma_f16_bf8_bf8_64_generic =
      DecodeVop3pStub("V_WMMA_F16_16X16X64_BF8_BF8_w32");
  if (!Expect(wmma_f16_bf8_bf8_64_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA F16 BF8/BF8 64 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f16_bf8_bf8_64_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 64, 8,
                                   0),
              "expected generic WMMA F16 BF8/BF8 64 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  wmma_f16_bf8_bf8_64_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 16),
              "expected generic WMMA F16 BF8/BF8 64 destination descriptor")) {
    return 1;
  }

  const StubDecodedInstruction wmma_f16_fp8_bf8_64_generic =
      DecodeVop3pStub("V_WMMA_F16_16X16X64_FP8_BF8_w32");
  if (!Expect(wmma_f16_fp8_bf8_64_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA F16 FP8/BF8 64 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f16_fp8_bf8_64_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 64, 8,
                                   0),
              "expected generic WMMA F16 FP8/BF8 64 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  wmma_f16_fp8_bf8_64_generic, StubOperandRole::kAccumulator,
                  StubOperandSlotKind::kAccumulatorSource,
                  StubOperandValueClass::kAccumulatorFragment,
                  StubOperandAccess::kRead, 1, StubFragmentKind::kMatrix, 16),
              "expected generic WMMA F16 FP8/BF8 64 accumulator descriptor")) {
    return 1;
  }

  const StubDecodedInstruction wmma_f16_fp8_bf8_generic =
      DecodeVop3pStub("V_WMMA_F16_16X16X128_FP8_BF8_w32");
  if (!Expect(wmma_f16_fp8_bf8_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA F16 FP8/BF8 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f16_fp8_bf8_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic WMMA F16 FP8/BF8 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f16_fp8_bf8_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic WMMA F16 FP8/BF8 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  wmma_f16_fp8_bf8_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 16),
              "expected generic WMMA F16 FP8/BF8 destination descriptor")) {
    return 1;
  }

  const StubDecodedInstruction wmma_f16_bf8_fp8_128_generic =
      DecodeVop3pStub("V_WMMA_F16_16X16X128_BF8_FP8_w32");
  if (!Expect(wmma_f16_bf8_fp8_128_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA F16 BF8/FP8 128 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f16_bf8_fp8_128_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic WMMA F16 BF8/FP8 128 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  wmma_f16_bf8_fp8_128_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 16),
              "expected generic WMMA F16 BF8/FP8 128 destination descriptor")) {
    return 1;
  }

  const StubDecodedInstruction wmma_f16_bf8_bf8_generic =
      DecodeVop3pStub("V_WMMA_F16_16X16X128_BF8_BF8_w32");
  if (!Expect(wmma_f16_bf8_bf8_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA F16 BF8/BF8 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f16_bf8_bf8_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic WMMA F16 BF8/BF8 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f16_bf8_bf8_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic WMMA F16 BF8/BF8 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  wmma_f16_bf8_bf8_generic, StubOperandRole::kAccumulator,
                  StubOperandSlotKind::kAccumulatorSource,
                  StubOperandValueClass::kAccumulatorFragment,
                  StubOperandAccess::kRead, 1, StubFragmentKind::kMatrix, 16),
              "expected generic WMMA F16 BF8/BF8 accumulator descriptor")) {
    return 1;
  }

  const StubDecodedInstruction wmma_f16_fp8_fp8_generic =
      DecodeVop3pStub("V_WMMA_F16_16X16X64_FP8_FP8_w32");
  if (!Expect(wmma_f16_fp8_fp8_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA F16 FP8/FP8 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f16_fp8_fp8_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 64, 8,
                                   0),
              "expected generic WMMA F16 FP8/FP8 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f16_fp8_fp8_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 64, 8,
                                   0),
              "expected generic WMMA F16 FP8/FP8 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  wmma_f16_fp8_fp8_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 16),
              "expected generic WMMA F16 FP8/FP8 destination descriptor")) {
    return 1;
  }

  const StubDecodedInstruction wmma_f32_bf8_bf8_generic =
      DecodeVop3pStub("V_WMMA_F32_16X16X128_BF8_BF8_w32");
  if (!Expect(wmma_f32_bf8_bf8_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA F32 BF8/BF8 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f32_bf8_bf8_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic WMMA F32 BF8/BF8 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  wmma_f32_bf8_bf8_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 32),
              "expected generic WMMA F32 BF8/BF8 destination descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f32_bf8_bf8_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic WMMA F32 BF8/BF8 source1 fragment shape")) {
    return 1;
  }

  const StubDecodedInstruction wmma_f32_bf8_fp8_generic =
      DecodeVop3pStub("V_WMMA_F32_16X16X128_BF8_FP8_w32");
  if (!Expect(wmma_f32_bf8_fp8_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA F32 BF8/FP8 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f32_bf8_fp8_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic WMMA F32 BF8/FP8 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f32_bf8_fp8_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic WMMA F32 BF8/FP8 source1 fragment shape")) {
    return 1;
  }

  const StubDecodedInstruction wmma_f32_fp8_bf8_128_generic =
      DecodeVop3pStub("V_WMMA_F32_16X16X128_FP8_BF8_w32");
  if (!Expect(wmma_f32_fp8_bf8_128_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA F32 FP8/BF8 128 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f32_fp8_bf8_128_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic WMMA F32 FP8/BF8 128 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  wmma_f32_fp8_bf8_128_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 32),
              "expected generic WMMA F32 FP8/BF8 128 destination descriptor")) {
    return 1;
  }

  const StubDecodedInstruction wmma_f32_bf8_fp8_64_generic =
      DecodeVop3pStub("V_WMMA_F32_16X16X64_BF8_FP8_w32");
  if (!Expect(wmma_f32_bf8_fp8_64_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA F32 BF8/FP8 64 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f32_bf8_fp8_64_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 64, 8,
                                   0),
              "expected generic WMMA F32 BF8/FP8 64 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  wmma_f32_bf8_fp8_64_generic, StubOperandRole::kAccumulator,
                  StubOperandSlotKind::kAccumulatorSource,
                  StubOperandValueClass::kAccumulatorFragment,
                  StubOperandAccess::kRead, 1, StubFragmentKind::kMatrix, 32),
              "expected generic WMMA F32 BF8/FP8 64 accumulator descriptor")) {
    return 1;
  }

  const StubDecodedInstruction wmma_f32_fp8_bf8_generic =
      DecodeVop3pStub("V_WMMA_F32_16X16X64_FP8_BF8_w32");
  if (!Expect(wmma_f32_fp8_bf8_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA F32 FP8/BF8 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f32_fp8_bf8_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 64, 8,
                                   0),
              "expected generic WMMA F32 FP8/BF8 source fragment shape")) {
    return 1;
  }

  const StubDecodedInstruction wmma_f32_bf8_bf8_64_generic =
      DecodeVop3pStub("V_WMMA_F32_16X16X64_BF8_BF8_w32");
  if (!Expect(wmma_f32_bf8_bf8_64_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA F32 BF8/BF8 64 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f32_bf8_bf8_64_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 64, 8,
                                   0),
              "expected generic WMMA F32 BF8/BF8 64 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  wmma_f32_bf8_bf8_64_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 32),
              "expected generic WMMA F32 BF8/BF8 64 destination descriptor")) {
    return 1;
  }

  const StubDecodedInstruction wmma_f32_f16_generic =
      DecodeVop3pStub("V_WMMA_F32_16X16X32_F16_w32");
  if (!Expect(wmma_f32_f16_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA F32 F16 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f32_f16_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 32, 16,
                                   0),
              "expected generic WMMA F32 F16 source fragment shape")) {
    return 1;
  }

  const StubDecodedInstruction wmma_f32_f4_core_generic =
      DecodeVop3pStub("V_WMMA_F32_32X16X128_F4_w32");
  if (!Expect(wmma_f32_f4_core_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaCoreGeneric,
              "expected generic WMMA F4 core operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(wmma_f32_f4_core_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 32, 16, 128, 4,
                                   0),
              "expected generic WMMA F4 core source fragment shape")) {
    return 1;
  }

  const StubDecodedInstruction swmmac_bf16_generic =
      DecodeVop3pStub("V_SWMMAC_BF16_16X16X64_BF16_w32");
  if (!Expect(swmmac_bf16_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kSwmmacCoreGeneric,
              "expected generic SWMMAC BF16 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_bf16_generic,
                                   StubOperandSlotKind::kDestination,
                                   StubFragmentKind::kMatrix, 16, 16, 64, 16,
                                   0),
              "expected generic SWMMAC BF16 destination fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_bf16_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 64, 16,
                                   0),
              "expected generic SWMMAC BF16 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  swmmac_bf16_generic, StubOperandRole::kAccumulator,
                  StubOperandSlotKind::kAccumulatorSource,
                  StubOperandValueClass::kAccumulatorFragment,
                  StubOperandAccess::kRead, 1, StubFragmentKind::kMatrix, 16),
              "expected generic SWMMAC BF16 accumulator descriptor")) {
    return 1;
  }

  const StubDecodedInstruction swmmac_f16_bf8_fp8_generic =
      DecodeVop3pStub("V_SWMMAC_F16_16X16X128_BF8_FP8_w32");
  if (!Expect(swmmac_f16_bf8_fp8_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kSwmmacCoreGeneric,
              "expected generic SWMMAC F16 BF8/FP8 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_f16_bf8_fp8_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic SWMMAC F16 BF8/FP8 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  swmmac_f16_bf8_fp8_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 16),
              "expected generic SWMMAC F16 BF8/FP8 destination descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_f16_bf8_fp8_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic SWMMAC F16 BF8/FP8 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  swmmac_f16_bf8_fp8_generic, StubOperandRole::kAccumulator,
                  StubOperandSlotKind::kAccumulatorSource,
                  StubOperandValueClass::kAccumulatorFragment,
                  StubOperandAccess::kRead, 1, StubFragmentKind::kMatrix, 16),
              "expected generic SWMMAC F16 BF8/FP8 accumulator descriptor")) {
    return 1;
  }

  const StubDecodedInstruction swmmac_f16_fp8_bf8_generic =
      DecodeVop3pStub("V_SWMMAC_F16_16X16X128_FP8_BF8_w32");
  if (!Expect(swmmac_f16_fp8_bf8_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kSwmmacCoreGeneric,
              "expected generic SWMMAC F16 FP8/BF8 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_f16_fp8_bf8_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic SWMMAC F16 FP8/BF8 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_f16_fp8_bf8_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic SWMMAC F16 FP8/BF8 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  swmmac_f16_fp8_bf8_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 16),
              "expected generic SWMMAC F16 FP8/BF8 destination descriptor")) {
    return 1;
  }

  const StubDecodedInstruction swmmac_f16_bf8_bf8_generic =
      DecodeVop3pStub("V_SWMMAC_F16_16X16X128_BF8_BF8_w32");
  if (!Expect(swmmac_f16_bf8_bf8_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kSwmmacCoreGeneric,
              "expected generic SWMMAC F16 BF8/BF8 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_f16_bf8_bf8_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic SWMMAC F16 BF8/BF8 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_f16_bf8_bf8_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic SWMMAC F16 BF8/BF8 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  swmmac_f16_bf8_bf8_generic, StubOperandRole::kAccumulator,
                  StubOperandSlotKind::kAccumulatorSource,
                  StubOperandValueClass::kAccumulatorFragment,
                  StubOperandAccess::kRead, 1, StubFragmentKind::kMatrix, 16),
              "expected generic SWMMAC F16 BF8/BF8 accumulator descriptor")) {
    return 1;
  }

  const StubDecodedInstruction swmmac_f32_bf8_bf8_generic =
      DecodeVop3pStub("V_SWMMAC_F32_16X16X128_BF8_BF8_w32");
  if (!Expect(swmmac_f32_bf8_bf8_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kSwmmacCoreGeneric,
              "expected generic SWMMAC F32 BF8/BF8 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_f32_bf8_bf8_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic SWMMAC F32 BF8/BF8 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_f32_bf8_bf8_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic SWMMAC F32 BF8/BF8 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  swmmac_f32_bf8_bf8_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 32),
              "expected generic SWMMAC F32 BF8/BF8 destination descriptor")) {
    return 1;
  }

  const StubDecodedInstruction swmmac_f32_bf8_fp8_generic =
      DecodeVop3pStub("V_SWMMAC_F32_16X16X128_BF8_FP8_w32");
  if (!Expect(swmmac_f32_bf8_fp8_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kSwmmacCoreGeneric,
              "expected generic SWMMAC F32 BF8/FP8 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_f32_bf8_fp8_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic SWMMAC F32 BF8/FP8 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_f32_bf8_fp8_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic SWMMAC F32 BF8/FP8 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  swmmac_f32_bf8_fp8_generic, StubOperandRole::kAccumulator,
                  StubOperandSlotKind::kAccumulatorSource,
                  StubOperandValueClass::kAccumulatorFragment,
                  StubOperandAccess::kRead, 1, StubFragmentKind::kMatrix, 32),
              "expected generic SWMMAC F32 BF8/FP8 accumulator descriptor")) {
    return 1;
  }

  const StubDecodedInstruction swmmac_f32_fp8_bf8_generic =
      DecodeVop3pStub("V_SWMMAC_F32_16X16X128_FP8_BF8_w32");
  if (!Expect(swmmac_f32_fp8_bf8_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kSwmmacCoreGeneric,
              "expected generic SWMMAC F32 FP8/BF8 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_f32_fp8_bf8_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic SWMMAC F32 FP8/BF8 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_f32_fp8_bf8_generic,
                                   StubOperandSlotKind::kSource1,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 8,
                                   0),
              "expected generic SWMMAC F32 FP8/BF8 source1 fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  swmmac_f32_fp8_bf8_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 32),
              "expected generic SWMMAC F32 FP8/BF8 destination descriptor")) {
    return 1;
  }

  const StubDecodedInstruction swmmac_f32_bf16_generic =
      DecodeVop3pStub("V_SWMMAC_F32_16X16X64_BF16_w32");
  if (!Expect(swmmac_f32_bf16_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kSwmmacCoreGeneric,
              "expected generic SWMMAC F32 BF16 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_f32_bf16_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 64, 16,
                                   0),
              "expected generic SWMMAC F32 BF16 source fragment shape")) {
    return 1;
  }

  const StubDecodedInstruction swmmac_f16_f16_generic =
      DecodeVop3pStub("V_SWMMAC_F16_16X16X64_F16_w32");
  if (!Expect(swmmac_f16_f16_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kSwmmacCoreGeneric,
              "expected generic SWMMAC F16/F16 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_f16_f16_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 64, 16,
                                   0),
              "expected generic SWMMAC F16/F16 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  swmmac_f16_f16_generic, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination,
                  StubOperandValueClass::kMatrixFragment,
                  StubOperandAccess::kWrite, 1, StubFragmentKind::kMatrix, 16),
              "expected generic SWMMAC F16/F16 destination descriptor")) {
    return 1;
  }

  const StubDecodedInstruction swmmac_f32_f16_generic =
      DecodeVop3pStub("V_SWMMAC_F32_16X16X64_F16_w32");
  if (!Expect(swmmac_f32_f16_generic.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kSwmmacCoreGeneric,
              "expected generic SWMMAC F32 F16 operand layout")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(swmmac_f32_f16_generic,
                                   StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kMatrix, 16, 16, 64, 16,
                                   0),
              "expected generic SWMMAC F32 F16 source fragment shape")) {
    return 1;
  }

  const StubDecodedInstruction tensor =
      DecodeMimgTensorStub("TENSOR_LOAD_TO_LDS");
  if (!Expect(tensor.status == StubDecodeStatus::kDecodedStub,
              "expected tensor load to decode through tensor stub")) {
    return 1;
  }
  if (!Expect(tensor.entrypoint_name == "DecodeMimgTensorStub",
              "expected tensor entrypoint name")) {
    return 1;
  }
  if (!Expect(tensor.opcode_shape == StubOpcodeShape::kTensorLoadToLds,
              "expected tensor-load shape")) {
    return 1;
  }
  if (!Expect(tensor.uses_tensor_memory,
              "expected tensor load to touch tensor-memory path")) {
    return 1;
  }
  if (!Expect(tensor.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kTensorLoadToLds,
              "expected tensor-load operand layout")) {
    return 1;
  }
  if (!Expect(tensor.operand_layout.has_tensor_descriptor &&
                  tensor.operand_layout.touches_lds &&
                  !tensor.operand_layout.is_store,
              "expected tensor-load operand layout flags")) {
    return 1;
  }
  if (!Expect(
          ContainsRole(tensor, StubOperandRole::kTensorDescriptor, 1, false) &&
              ContainsRole(tensor, StubOperandRole::kTensorCoordinate, 1, false) &&
              ContainsRole(tensor, StubOperandRole::kLdsDestination, 1, true),
          "expected tensor-load operand roles")) {
    return 1;
  }
  if (!Expect(
          ContainsSlot(tensor, StubOperandSlotKind::kTensorDescriptorSource,
                       StubOperandValueClass::kTensorDescriptor, 0, 1, false) &&
              ContainsSlot(tensor,
                           StubOperandSlotKind::kTensorCoordinateSource,
                           StubOperandValueClass::kTensorCoordinate, 1, 1,
                           false) &&
              ContainsSlot(tensor, StubOperandSlotKind::kLdsDestination,
                           StubOperandValueClass::kLdsAddress, 2, 1, true),
          "expected tensor-load operand slots")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(
                  tensor, StubOperandSlotKind::kTensorDescriptorSource,
                  StubFragmentKind::kTensorDescriptor, 1, 1, 1, 0, 1),
              "expected tensor-load descriptor fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  tensor, StubOperandRole::kTensorDescriptor,
                  StubOperandSlotKind::kTensorDescriptorSource,
                  StubOperandValueClass::kTensorDescriptor,
                  StubOperandAccess::kRead, 1,
                  StubFragmentKind::kTensorDescriptor, 0),
              "expected tensor-load tensor-descriptor descriptor")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  tensor, StubOperandRole::kTensorCoordinate,
                  StubOperandSlotKind::kTensorCoordinateSource,
                  StubOperandValueClass::kTensorCoordinate,
                  StubOperandAccess::kRead, 1,
                  StubFragmentKind::kTensorCoordinate, 0),
              "expected tensor-load tensor-coordinate descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(
                  tensor, StubOperandSlotKind::kTensorCoordinateSource,
                  StubFragmentKind::kTensorCoordinate, 1, 1, 1, 0, 1),
              "expected tensor-load coordinate fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  tensor, StubOperandRole::kLdsDestination,
                  StubOperandSlotKind::kLdsDestination,
                  StubOperandValueClass::kLdsAddress, StubOperandAccess::kWrite,
                  1, StubFragmentKind::kAddress, 32),
              "expected tensor-load LDS destination descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(tensor, StubOperandSlotKind::kLdsDestination,
                                   StubFragmentKind::kAddress, 1, 1, 1, 32, 1),
              "expected tensor-load LDS destination fragment shape")) {
    return 1;
  }

  const StubDecodedInstruction tensor_store =
      DecodeMimgTensorStub("TENSOR_STORE_FROM_LDS");
  if (!Expect(tensor_store.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kTensorStoreFromLds,
              "expected tensor-store operand layout")) {
    return 1;
  }
  if (!Expect(tensor_store.operand_layout.has_tensor_descriptor &&
                  tensor_store.operand_layout.touches_lds &&
                  tensor_store.operand_layout.is_store,
              "expected tensor-store operand layout flags")) {
    return 1;
  }
  if (!Expect(
          ContainsRole(tensor_store, StubOperandRole::kTensorDescriptor, 1, false) &&
              ContainsRole(tensor_store, StubOperandRole::kTensorCoordinate, 1, false) &&
              ContainsRole(tensor_store, StubOperandRole::kLdsSource, 1, false),
          "expected tensor-store operand roles")) {
    return 1;
  }
  if (!Expect(
          ContainsSlot(tensor_store,
                       StubOperandSlotKind::kTensorDescriptorSource,
                       StubOperandValueClass::kTensorDescriptor, 0, 1, false) &&
              ContainsSlot(tensor_store,
                           StubOperandSlotKind::kTensorCoordinateSource,
                           StubOperandValueClass::kTensorCoordinate, 1, 1,
                           false) &&
              ContainsSlot(tensor_store, StubOperandSlotKind::kLdsSource,
                           StubOperandValueClass::kLdsAddress, 2, 1, false),
          "expected tensor-store operand slots")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  tensor_store, StubOperandRole::kLdsSource,
                  StubOperandSlotKind::kLdsSource,
                  StubOperandValueClass::kLdsAddress, StubOperandAccess::kRead,
                  1, StubFragmentKind::kAddress, 32),
              "expected tensor-store LDS source descriptor")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  tensor_store, StubOperandRole::kTensorDescriptor,
                  StubOperandSlotKind::kTensorDescriptorSource,
                  StubOperandValueClass::kTensorDescriptor,
                  StubOperandAccess::kRead, 1,
                  StubFragmentKind::kTensorDescriptor, 0),
              "expected tensor-store tensor-descriptor descriptor")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(
                  tensor_store, StubOperandRole::kTensorCoordinate,
                  StubOperandSlotKind::kTensorCoordinateSource,
                  StubOperandValueClass::kTensorCoordinate,
                  StubOperandAccess::kRead, 1,
                  StubFragmentKind::kTensorCoordinate, 0),
              "expected tensor-store tensor-coordinate descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(
                  tensor_store, StubOperandSlotKind::kTensorCoordinateSource,
                  StubFragmentKind::kTensorCoordinate, 1, 1, 1, 0, 1),
              "expected tensor-store coordinate fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(tensor_store, StubOperandSlotKind::kLdsSource,
                                   StubFragmentKind::kAddress, 1, 1, 1, 32, 1),
              "expected tensor-store LDS source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(
                  tensor_store, StubOperandSlotKind::kTensorDescriptorSource,
                  StubFragmentKind::kTensorDescriptor, 1, 1, 1, 0, 1),
              "expected tensor-store descriptor fragment shape")) {
    return 1;
  }

  const StubDecodedInstruction vop1 = DecodeVop1Stub("V_CVT_F16_FP8");
  if (!Expect(vop1.status == StubDecodeStatus::kDecodedStub,
              "expected FP8 conversion to decode through VOP1 stub")) {
    return 1;
  }
  if (!Expect(vop1.route_priority == 3,
              "expected VOP1 stub priority to be preserved")) {
    return 1;
  }
  if (!Expect(vop1.opcode_shape == StubOpcodeShape::kFp8ConvertToF16,
              "expected F16 conversion shape for V_CVT_F16_FP8")) {
    return 1;
  }
  if (!Expect(vop1.execution_domain == StubExecutionDomain::kConversion,
              "expected conversion domain for VOP1 FP8 seed")) {
    return 1;
  }
  if (!Expect(vop1.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kCvtF16Fp8,
              "expected explicit operand layout for V_CVT_F16_FP8")) {
    return 1;
  }
  if (!Expect(vop1.operand_layout.source_count == 1 &&
                  vop1.operand_layout.destination_count == 1,
              "expected V_CVT_F16_FP8 operand layout counts")) {
    return 1;
  }
  if (!Expect(ContainsRole(vop1, StubOperandRole::kSource0, 1, false) &&
                  ContainsRole(vop1, StubOperandRole::kDestination, 1, true),
              "expected V_CVT_F16_FP8 operand roles")) {
    return 1;
  }
  if (!Expect(ContainsSlot(vop1, StubOperandSlotKind::kDestination,
                           StubOperandValueClass::kVectorRegister, 0, 1, true) &&
                  ContainsSlot(vop1, StubOperandSlotKind::kSource0,
                               StubOperandValueClass::kVectorRegister, 1, 1,
                               false),
              "expected V_CVT_F16_FP8 operand slots")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(vop1, StubOperandRole::kSource0,
                                 StubOperandSlotKind::kSource0,
                                 StubOperandValueClass::kVectorRegister,
                                 StubOperandAccess::kRead, 1,
                                 StubFragmentKind::kScalar, 8),
              "expected V_CVT_F16_FP8 source descriptor")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(vop1, StubOperandRole::kDestination,
                                 StubOperandSlotKind::kDestination,
                                 StubOperandValueClass::kVectorRegister,
                                 StubOperandAccess::kWrite, 1,
                                 StubFragmentKind::kScalar, 16),
              "expected V_CVT_F16_FP8 destination descriptor")) {
    return 1;
  }

  const StubDecodedInstruction vop1_bf8 = DecodeVop1Stub("V_CVT_F16_BF8");
  if (!Expect(vop1_bf8.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kCvtF16Bf8,
              "expected explicit operand layout for V_CVT_F16_BF8")) {
    return 1;
  }
  if (!Expect(ContainsSlot(vop1_bf8, StubOperandSlotKind::kSource0,
                           StubOperandValueClass::kVectorRegister, 1, 1, false),
              "expected V_CVT_F16_BF8 source slot")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(vop1_bf8, StubOperandRole::kDestination,
                                 StubOperandSlotKind::kDestination,
                                 StubOperandValueClass::kVectorRegister,
                                 StubOperandAccess::kWrite, 1,
                                 StubFragmentKind::kScalar, 16),
              "expected V_CVT_F16_BF8 destination descriptor")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(vop1_bf8, StubOperandRole::kSource0,
                                 StubOperandSlotKind::kSource0,
                                 StubOperandValueClass::kVectorRegister,
                                 StubOperandAccess::kRead, 1,
                                 StubFragmentKind::kScalar, 8),
              "expected V_CVT_F16_BF8 source descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(vop1_bf8, StubOperandSlotKind::kDestination,
                                   StubFragmentKind::kScalar, 1, 1, 1, 16, 1),
              "expected V_CVT_F16_BF8 destination fragment shape")) {
    return 1;
  }

  const StubDecodedInstruction vop1_f32 = DecodeVop1Stub("V_CVT_F32_FP8");
  if (!Expect(vop1_f32.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kCvtF32Fp8,
              "expected explicit operand layout for V_CVT_F32_FP8")) {
    return 1;
  }
  if (!Expect(vop1_f32.operand_layout.source_count == 1 &&
                  vop1_f32.operand_layout.destination_count == 1,
              "expected V_CVT_F32_FP8 operand layout counts")) {
    return 1;
  }
  if (!Expect(ContainsRole(vop1_f32, StubOperandRole::kSource0, 1, false) &&
                  ContainsRole(vop1_f32, StubOperandRole::kDestination, 1, true),
              "expected V_CVT_F32_FP8 operand roles")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(vop1_f32, StubOperandSlotKind::kSource0,
                                   StubFragmentKind::kScalar, 1, 1, 1, 8, 1),
              "expected V_CVT_F32_FP8 source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(vop1_f32, StubOperandRole::kDestination,
                                 StubOperandSlotKind::kDestination,
                                 StubOperandValueClass::kVectorRegister,
                                 StubOperandAccess::kWrite, 1,
                                 StubFragmentKind::kScalar, 32),
              "expected V_CVT_F32_FP8 destination descriptor")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(vop1_f32, StubOperandRole::kSource0,
                                 StubOperandSlotKind::kSource0,
                                 StubOperandValueClass::kVectorRegister,
                                 StubOperandAccess::kRead, 1,
                                 StubFragmentKind::kScalar, 8),
              "expected V_CVT_F32_FP8 source descriptor")) {
    return 1;
  }

  const StubDecodedInstruction packed_vop1 =
      DecodeVop1Stub("V_CVT_PK_F16_FP8");
  if (!Expect(packed_vop1.opcode_shape == StubOpcodeShape::kFp8PackedConvert,
              "expected packed-conversion shape for V_CVT_PK_F16_FP8")) {
    return 1;
  }
  if (!Expect(packed_vop1.uses_paired_operands,
              "expected packed-conversion shape to use paired operands")) {
    return 1;
  }
  if (!Expect(packed_vop1.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kCvtPkF16Fp8,
              "expected explicit operand layout for V_CVT_PK_F16_FP8")) {
    return 1;
  }
  if (!Expect(ContainsSlot(packed_vop1, StubOperandSlotKind::kDestination,
                           StubOperandValueClass::kPackedVector, 0, 2, true),
              "expected packed FP8 conversion destination slot")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(
                  packed_vop1, StubOperandSlotKind::kSource0,
                  StubFragmentKind::kPacked, 1, 1, 1, 8, 2),
              "expected packed FP8 conversion source fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(packed_vop1, StubOperandRole::kDestination,
                                 StubOperandSlotKind::kDestination,
                                 StubOperandValueClass::kPackedVector,
                                 StubOperandAccess::kWrite, 2,
                                 StubFragmentKind::kPacked, 16),
              "expected packed FP8 conversion destination descriptor")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(packed_vop1, StubOperandRole::kSource0,
                                 StubOperandSlotKind::kSource0,
                                 StubOperandValueClass::kPackedVector,
                                 StubOperandAccess::kRead, 2,
                                 StubFragmentKind::kPacked, 8),
              "expected packed FP8 conversion source descriptor")) {
    return 1;
  }

  const StubDecodedInstruction packed_bf8 =
      DecodeVop1Stub("V_CVT_PK_F16_BF8");
  if (!Expect(packed_bf8.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kCvtPkF16Bf8,
              "expected explicit operand layout for V_CVT_PK_F16_BF8")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(packed_bf8, StubOperandRole::kSource0,
                                 StubOperandSlotKind::kSource0,
                                 StubOperandValueClass::kPackedVector,
                                 StubOperandAccess::kRead, 2,
                                 StubFragmentKind::kPacked, 8),
              "expected packed BF8 conversion source descriptor")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(packed_bf8, StubOperandRole::kDestination,
                                 StubOperandSlotKind::kDestination,
                                 StubOperandValueClass::kPackedVector,
                                 StubOperandAccess::kWrite, 2,
                                 StubFragmentKind::kPacked, 16),
              "expected packed BF8 conversion destination descriptor")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(
                  packed_bf8, StubOperandSlotKind::kDestination,
                  StubFragmentKind::kPacked, 1, 1, 1, 16, 2),
              "expected packed BF8 conversion destination fragment shape")) {
    return 1;
  }

  const StubDecodedInstruction sdst = DecodeVop3SdstStub("V_DIV_SCALE_F64");
  if (!Expect(sdst.status == StubDecodeStatus::kDecodedStub,
              "expected V_DIV_SCALE_F64 to decode through VOP3 SDST stub")) {
    return 1;
  }
  if (!Expect(sdst.rdna4_encoding_name == "VOP3_SDST_ENC",
              "expected SDST stub to preserve RDNA4 encoding name")) {
    return 1;
  }
  if (!Expect(sdst.opcode_shape == StubOpcodeShape::kVop3SdstScale,
              "expected VOP3 SDST scale shape")) {
    return 1;
  }
  if (!Expect(sdst.execution_domain == StubExecutionDomain::kScaleAssist,
              "expected scale-assist domain for V_DIV_SCALE_F64")) {
    return 1;
  }
  if (!Expect(sdst.uses_scale_path,
              "expected VOP3 SDST scale path flag")) {
    return 1;
  }
  if (!Expect(sdst.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kVDivScaleF64,
              "expected explicit operand layout for V_DIV_SCALE_F64")) {
    return 1;
  }
  if (!Expect(sdst.operand_layout.source_count == 3 &&
                  sdst.operand_layout.destination_count == 2 &&
                  sdst.operand_layout.has_scale_operand,
              "expected V_DIV_SCALE_F64 operand layout counts and flags")) {
    return 1;
  }
  if (!Expect(ContainsRole(sdst, StubOperandRole::kSource0, 1, false) &&
                  ContainsRole(sdst, StubOperandRole::kSource1, 1, false) &&
                  ContainsRole(sdst, StubOperandRole::kScale, 1, false) &&
                  ContainsRole(sdst, StubOperandRole::kDestination, 1, true),
              "expected V_DIV_SCALE_F64 operand roles")) {
    return 1;
  }
  if (!Expect(
          ContainsSlot(sdst, StubOperandSlotKind::kDestination,
                       StubOperandValueClass::kVectorRegister, 0, 2, true) &&
              ContainsSlot(sdst,
                           StubOperandSlotKind::kScalarDestination,
                           StubOperandValueClass::kScalarRegister, 1, 1,
                           true) &&
              ContainsSlot(sdst, StubOperandSlotKind::kSource0,
                           StubOperandValueClass::kVectorRegister, 2, 2,
                           false) &&
              ContainsSlot(sdst, StubOperandSlotKind::kSource1,
                           StubOperandValueClass::kVectorRegister, 3, 2,
                           false) &&
              ContainsSlot(sdst, StubOperandSlotKind::kScaleSource,
                           StubOperandValueClass::kVectorRegister, 4, 2,
                           false),
          "expected V_DIV_SCALE_F64 operand slots")) {
    return 1;
  }
  if (!Expect(ContainsSlotFragment(sdst, StubOperandSlotKind::kDestination,
                                   StubFragmentKind::kScalar, 1, 1, 1, 64, 1),
              "expected V_DIV_SCALE_F64 vector destination fragment shape")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(sdst, StubOperandRole::kDestination,
                                 StubOperandSlotKind::kScalarDestination,
                                 StubOperandValueClass::kScalarRegister,
                                 StubOperandAccess::kWrite, 1,
                                 StubFragmentKind::kScalar, 32),
              "expected V_DIV_SCALE_F64 scalar destination descriptor")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(sdst, StubOperandRole::kScale,
                                 StubOperandSlotKind::kScaleSource,
                                 StubOperandValueClass::kVectorRegister,
                                 StubOperandAccess::kRead, 2,
                                 StubFragmentKind::kScalar, 64),
              "expected V_DIV_SCALE_F64 scale-source descriptor")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(sdst, StubOperandRole::kSource0,
                                 StubOperandSlotKind::kSource0,
                                 StubOperandValueClass::kVectorRegister,
                                 StubOperandAccess::kRead, 2,
                                 StubFragmentKind::kScalar, 64),
              "expected V_DIV_SCALE_F64 source0 descriptor")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(sdst, StubOperandRole::kSource1,
                                 StubOperandSlotKind::kSource1,
                                 StubOperandValueClass::kVectorRegister,
                                 StubOperandAccess::kRead, 2,
                                 StubFragmentKind::kScalar, 64),
              "expected V_DIV_SCALE_F64 source1 descriptor")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(sdst, StubOperandRole::kDestination,
                                 StubOperandSlotKind::kDestination,
                                 StubOperandValueClass::kVectorRegister,
                                 StubOperandAccess::kWrite, 2,
                                 StubFragmentKind::kScalar, 64),
              "expected V_DIV_SCALE_F64 vector destination descriptor")) {
    return 1;
  }
  if (!Expect(ContainsDescriptor(paired_scale16, StubOperandRole::kDestination,
                                 StubOperandSlotKind::kDestination,
                                 StubOperandValueClass::kVectorRegister,
                                 StubOperandAccess::kWrite, 1,
                                 StubFragmentKind::kVector, 64),
              "expected paired WMMA scale16 destination descriptor")) {
    return 1;
  }

  for (std::string_view instruction_name :
       GetStubDecoderRouteInstructions(StubDecoderRoute::kMimgTensor)) {
    const StubDecodedInstruction decoded = DecodeMimgTensorStub(instruction_name);
    if (!Expect(decoded.status == StubDecodeStatus::kDecodedStub,
                "expected routed tensor seed to decode")) {
      return 1;
    }
    if (!Expect(decoded.uses_tensor_memory &&
                    decoded.operand_layout.has_tensor_descriptor &&
                    decoded.operand_layout.touches_lds &&
                    decoded.operand_slots.binding_count == 3 &&
                    decoded.operand_descriptors.descriptor_count == 3 &&
                    CountSlotsWithFragmentKindAndWaveSize(
                        decoded, StubFragmentKind::kTensorDescriptor, 0) == 1 &&
                    CountSlotsWithFragmentKindAndWaveSize(
                        decoded, StubFragmentKind::kTensorCoordinate, 0) == 1 &&
                    CountSlotsWithFragmentKindAndWaveSize(
                        decoded, StubFragmentKind::kAddress, 0) == 1 &&
                    CountDescriptorsWithFragmentKindAndWaveSize(
                        decoded, StubFragmentKind::kTensorDescriptor, 0) == 1 &&
                    CountDescriptorsWithFragmentKindAndWaveSize(
                        decoded, StubFragmentKind::kTensorCoordinate, 0) == 1 &&
                    CountDescriptorsWithFragmentKindAndWaveSize(
                        decoded, StubFragmentKind::kAddress, 0) == 1 &&
                    CountOutputSlots(decoded) ==
                        (instruction_name == "TENSOR_LOAD_TO_LDS" ? 1u : 0u) &&
                    CountDescriptorsWithAccess(decoded, StubOperandAccess::kRead) ==
                        (instruction_name == "TENSOR_LOAD_TO_LDS" ? 2u : 3u) &&
                    CountDescriptorsWithAccess(decoded, StubOperandAccess::kWrite) ==
                        (instruction_name == "TENSOR_LOAD_TO_LDS" ? 1u : 0u) &&
                    !decoded.uses_scale_path && !decoded.uses_paired_operands &&
                    AllSlotsExplicit(decoded) &&
                    AllDescriptorsExplicit(decoded) &&
                    AllSlotWaveSizesAre(decoded, 0) &&
                    AllDescriptorWaveSizesAre(decoded, 0),
                "expected routed tensor seed to keep non-matrix wave-size semantics")) {
      return 1;
    }
    if (!Expect(CountDescriptorsForRole(decoded, StubOperandRole::kTensorDescriptor) == 1 &&
                    CountDescriptorsForRole(decoded, StubOperandRole::kTensorCoordinate) == 1 &&
                    (CountDescriptorsForRole(decoded, StubOperandRole::kLdsDestination) == 1 ||
                     CountDescriptorsForRole(decoded, StubOperandRole::kLdsSource) == 1) &&
                    CountSlotsOfKind(decoded, StubOperandSlotKind::kTensorDescriptorSource) == 1 &&
                    CountSlotsOfKind(decoded, StubOperandSlotKind::kTensorCoordinateSource) == 1,
                "expected routed tensor seed to keep exact descriptor/slot composition")) {
      return 1;
    }
    if (instruction_name == "TENSOR_LOAD_TO_LDS") {
      if (!Expect(HasDescriptorRole(decoded, StubOperandRole::kTensorDescriptor) &&
                      HasDescriptorRole(decoded, StubOperandRole::kTensorCoordinate) &&
                      HasDescriptorRole(decoded, StubOperandRole::kLdsDestination) &&
                      ContainsSlot(decoded,
                                   StubOperandSlotKind::kTensorDescriptorSource,
                                   StubOperandValueClass::kTensorDescriptor, 0,
                                   1, false) &&
                      ContainsSlot(decoded,
                                   StubOperandSlotKind::kTensorCoordinateSource,
                                   StubOperandValueClass::kTensorCoordinate, 1,
                                   1, false) &&
                      ContainsSlot(decoded, StubOperandSlotKind::kLdsDestination,
                                   StubOperandValueClass::kLdsAddress, 2, 1,
                                   true) &&
                      ContainsSlotFragment(
                          decoded, StubOperandSlotKind::kTensorDescriptorSource,
                          StubFragmentKind::kTensorDescriptor, 1, 1, 1, 0, 1) &&
                      ContainsSlotWaveSize(
                          decoded, StubOperandSlotKind::kTensorDescriptorSource,
                          StubFragmentKind::kTensorDescriptor, 0) &&
                      ContainsSlotFragment(
                          decoded, StubOperandSlotKind::kTensorCoordinateSource,
                          StubFragmentKind::kTensorCoordinate, 1, 1, 1, 0, 1) &&
                      ContainsSlotWaveSize(
                          decoded, StubOperandSlotKind::kTensorCoordinateSource,
                          StubFragmentKind::kTensorCoordinate, 0) &&
                      ContainsSlotFragment(decoded,
                                           StubOperandSlotKind::kLdsDestination,
                                           StubFragmentKind::kAddress, 1, 1, 1,
                                           32, 1) &&
                      ContainsSlotWaveSize(decoded,
                                           StubOperandSlotKind::kLdsDestination,
                                           StubFragmentKind::kAddress, 0) &&
                      ContainsDescriptor(decoded,
                                         StubOperandRole::kTensorDescriptor,
                                         StubOperandSlotKind::kTensorDescriptorSource,
                                         StubOperandValueClass::kTensorDescriptor,
                                         StubOperandAccess::kRead, 1,
                                         StubFragmentKind::kTensorDescriptor, 0) &&
                      ContainsDescriptorWaveSize(
                          decoded, StubOperandRole::kTensorDescriptor,
                          StubOperandSlotKind::kTensorDescriptorSource,
                          StubFragmentKind::kTensorDescriptor, 0) &&
                      ContainsDescriptor(decoded,
                                         StubOperandRole::kTensorCoordinate,
                                         StubOperandSlotKind::kTensorCoordinateSource,
                                         StubOperandValueClass::kTensorCoordinate,
                                         StubOperandAccess::kRead, 1,
                                         StubFragmentKind::kTensorCoordinate, 0) &&
                      ContainsDescriptorWaveSize(
                          decoded, StubOperandRole::kTensorCoordinate,
                          StubOperandSlotKind::kTensorCoordinateSource,
                          StubFragmentKind::kTensorCoordinate, 0) &&
                      ContainsDescriptor(decoded,
                                         StubOperandRole::kLdsDestination,
                                         StubOperandSlotKind::kLdsDestination,
                                         StubOperandValueClass::kLdsAddress,
                                         StubOperandAccess::kWrite, 1,
                                         StubFragmentKind::kAddress, 32) &&
                      ContainsDescriptorWaveSize(
                          decoded, StubOperandRole::kLdsDestination,
                          StubOperandSlotKind::kLdsDestination,
                          StubFragmentKind::kAddress, 0) &&
                      CountSlotsOfKind(decoded, StubOperandSlotKind::kLdsDestination) == 1,
                  "expected routed tensor-load descriptor roles")) {
        return 1;
      }
    } else if (instruction_name == "TENSOR_STORE_FROM_LDS") {
      if (!Expect(HasDescriptorRole(decoded, StubOperandRole::kTensorDescriptor) &&
                      HasDescriptorRole(decoded, StubOperandRole::kTensorCoordinate) &&
                      HasDescriptorRole(decoded, StubOperandRole::kLdsSource) &&
                      ContainsSlot(decoded,
                                   StubOperandSlotKind::kTensorDescriptorSource,
                                   StubOperandValueClass::kTensorDescriptor, 0,
                                   1, false) &&
                      ContainsSlot(decoded,
                                   StubOperandSlotKind::kTensorCoordinateSource,
                                   StubOperandValueClass::kTensorCoordinate, 1,
                                   1, false) &&
                      ContainsSlot(decoded, StubOperandSlotKind::kLdsSource,
                                   StubOperandValueClass::kLdsAddress, 2, 1,
                                   false) &&
                      ContainsSlotFragment(
                          decoded, StubOperandSlotKind::kTensorDescriptorSource,
                          StubFragmentKind::kTensorDescriptor, 1, 1, 1, 0, 1) &&
                      ContainsSlotWaveSize(
                          decoded, StubOperandSlotKind::kTensorDescriptorSource,
                          StubFragmentKind::kTensorDescriptor, 0) &&
                      ContainsSlotFragment(
                          decoded, StubOperandSlotKind::kTensorCoordinateSource,
                          StubFragmentKind::kTensorCoordinate, 1, 1, 1, 0, 1) &&
                      ContainsSlotWaveSize(
                          decoded, StubOperandSlotKind::kTensorCoordinateSource,
                          StubFragmentKind::kTensorCoordinate, 0) &&
                      ContainsSlotFragment(decoded,
                                           StubOperandSlotKind::kLdsSource,
                                           StubFragmentKind::kAddress, 1, 1, 1,
                                           32, 1) &&
                      ContainsSlotWaveSize(decoded,
                                           StubOperandSlotKind::kLdsSource,
                                           StubFragmentKind::kAddress, 0) &&
                      ContainsDescriptor(decoded,
                                         StubOperandRole::kTensorDescriptor,
                                         StubOperandSlotKind::kTensorDescriptorSource,
                                         StubOperandValueClass::kTensorDescriptor,
                                         StubOperandAccess::kRead, 1,
                                         StubFragmentKind::kTensorDescriptor, 0) &&
                      ContainsDescriptorWaveSize(
                          decoded, StubOperandRole::kTensorDescriptor,
                          StubOperandSlotKind::kTensorDescriptorSource,
                          StubFragmentKind::kTensorDescriptor, 0) &&
                      ContainsDescriptor(decoded,
                                         StubOperandRole::kTensorCoordinate,
                                         StubOperandSlotKind::kTensorCoordinateSource,
                                         StubOperandValueClass::kTensorCoordinate,
                                         StubOperandAccess::kRead, 1,
                                         StubFragmentKind::kTensorCoordinate, 0) &&
                      ContainsDescriptorWaveSize(
                          decoded, StubOperandRole::kTensorCoordinate,
                          StubOperandSlotKind::kTensorCoordinateSource,
                          StubFragmentKind::kTensorCoordinate, 0) &&
                      ContainsDescriptor(decoded,
                                         StubOperandRole::kLdsSource,
                                         StubOperandSlotKind::kLdsSource,
                                         StubOperandValueClass::kLdsAddress,
                                         StubOperandAccess::kRead, 1,
                                         StubFragmentKind::kAddress, 32) &&
                      ContainsDescriptorWaveSize(decoded,
                                                 StubOperandRole::kLdsSource,
                                                 StubOperandSlotKind::kLdsSource,
                                                 StubFragmentKind::kAddress, 0) &&
                      CountSlotsOfKind(decoded, StubOperandSlotKind::kLdsSource) == 1,
                  "expected routed tensor-store descriptor roles")) {
        return 1;
      }
    }
  }

  for (std::string_view instruction_name :
       GetStubDecoderRouteInstructions(StubDecoderRoute::kVop1)) {
    const StubDecodedInstruction decoded = DecodeVop1Stub(instruction_name);
    if (!Expect(decoded.status == StubDecodeStatus::kDecodedStub,
                "expected routed VOP1 seed to decode")) {
      return 1;
    }
    if (!Expect(decoded.execution_domain == StubExecutionDomain::kConversion &&
                    !HasMatrixSlot(decoded) && !HasMatrixDescriptor(decoded) &&
                    decoded.operand_slots.binding_count == 2 &&
                    decoded.operand_descriptors.descriptor_count == 2 &&
                    CountSlotsWithFragmentKindAndWaveSize(
                        decoded,
                        instruction_name.find("PK_") != std::string_view::npos
                            ? StubFragmentKind::kPacked
                            : StubFragmentKind::kScalar,
                        0) == 2 &&
                    CountDescriptorsWithFragmentKindAndWaveSize(
                        decoded,
                        instruction_name.find("PK_") != std::string_view::npos
                            ? StubFragmentKind::kPacked
                            : StubFragmentKind::kScalar,
                        0) == 2 &&
                    CountOutputSlots(decoded) == 1 &&
                    CountDescriptorsWithAccess(decoded, StubOperandAccess::kRead) == 1 &&
                    CountDescriptorsWithAccess(decoded, StubOperandAccess::kWrite) == 1 &&
                    !decoded.uses_scale_path && !decoded.uses_tensor_memory &&
                    AllSlotsExplicit(decoded) &&
                    AllDescriptorsExplicit(decoded) &&
                    AllSlotWaveSizesAre(decoded, 0) &&
                    AllDescriptorWaveSizesAre(decoded, 0),
                "expected routed VOP1 seed to keep scalar/packed wave-size semantics")) {
      return 1;
    }
    if (!Expect(HasDescriptorRole(decoded, StubOperandRole::kSource0) &&
                    HasDescriptorRole(decoded, StubOperandRole::kDestination) &&
                    CountDescriptorsForRole(decoded, StubOperandRole::kSource0) == 1 &&
                    CountDescriptorsForRole(decoded, StubOperandRole::kDestination) == 1 &&
                    CountSlotsOfKind(decoded, StubOperandSlotKind::kSource0) == 1 &&
                    CountSlotsOfKind(decoded, StubOperandSlotKind::kDestination) == 1,
                "expected routed VOP1 seed to keep source/destination descriptors")) {
      return 1;
    }
    if (instruction_name.find("PK_") != std::string_view::npos) {
      if (!Expect(
              ContainsSlot(decoded, StubOperandSlotKind::kSource0,
                           StubOperandValueClass::kPackedVector, 1, 2, false) &&
                  ContainsSlot(decoded, StubOperandSlotKind::kDestination,
                               StubOperandValueClass::kPackedVector, 0, 2,
                               true) &&
                  ContainsSlotFragment(decoded, StubOperandSlotKind::kSource0,
                                       StubFragmentKind::kPacked, 1, 1, 1, 8,
                                       2) &&
                  ContainsSlotWaveSize(decoded, StubOperandSlotKind::kSource0,
                                       StubFragmentKind::kPacked, 0) &&
                  ContainsSlotFragment(decoded,
                                       StubOperandSlotKind::kDestination,
                                       StubFragmentKind::kPacked, 1, 1, 1, 16,
                                       2) &&
                  ContainsSlotWaveSize(decoded,
                                       StubOperandSlotKind::kDestination,
                                       StubFragmentKind::kPacked, 0) &&
              ContainsDescriptor(decoded, StubOperandRole::kSource0,
                                 StubOperandSlotKind::kSource0,
                                 StubOperandValueClass::kPackedVector,
                                 StubOperandAccess::kRead, 2,
                                 StubFragmentKind::kPacked, 8) &&
                  ContainsDescriptorWaveSize(decoded, StubOperandRole::kSource0,
                                             StubOperandSlotKind::kSource0,
                                             StubFragmentKind::kPacked, 0) &&
                  ContainsDescriptor(decoded, StubOperandRole::kDestination,
                                     StubOperandSlotKind::kDestination,
                                     StubOperandValueClass::kPackedVector,
                                     StubOperandAccess::kWrite, 2,
                                     StubFragmentKind::kPacked, 16) &&
                  ContainsDescriptorWaveSize(
                      decoded, StubOperandRole::kDestination,
                      StubOperandSlotKind::kDestination,
                      StubFragmentKind::kPacked, 0),
              "expected routed packed VOP1 seed to keep packed descriptor shapes")) {
        return 1;
      }
      if (!Expect(decoded.uses_paired_operands,
                  "expected routed packed VOP1 seed to preserve paired-operand flag")) {
        return 1;
      }
    } else {
      if (!Expect(
              ContainsSlot(decoded, StubOperandSlotKind::kSource0,
                           StubOperandValueClass::kVectorRegister, 1, 1,
                           false) &&
                  ContainsSlot(decoded, StubOperandSlotKind::kDestination,
                               StubOperandValueClass::kVectorRegister, 0, 1,
                               true) &&
                  ContainsSlotFragment(decoded, StubOperandSlotKind::kSource0,
                                       StubFragmentKind::kScalar, 1, 1, 1, 8,
                                       1) &&
                  ContainsSlotWaveSize(decoded, StubOperandSlotKind::kSource0,
                                       StubFragmentKind::kScalar, 0) &&
                  ContainsSlotFragment(
                      decoded, StubOperandSlotKind::kDestination,
                      StubFragmentKind::kScalar, 1, 1, 1,
                      instruction_name == "V_CVT_F32_FP8" ? 32 : 16, 1) &&
                  ContainsSlotWaveSize(decoded,
                                       StubOperandSlotKind::kDestination,
                                       StubFragmentKind::kScalar, 0) &&
              ContainsDescriptor(decoded, StubOperandRole::kSource0,
                                 StubOperandSlotKind::kSource0,
                                 StubOperandValueClass::kVectorRegister,
                                 StubOperandAccess::kRead, 1,
                                 StubFragmentKind::kScalar, 8) &&
                  ContainsDescriptorWaveSize(decoded, StubOperandRole::kSource0,
                                             StubOperandSlotKind::kSource0,
                                             StubFragmentKind::kScalar, 0) &&
                  ContainsDescriptor(
                      decoded, StubOperandRole::kDestination,
                      StubOperandSlotKind::kDestination,
                      StubOperandValueClass::kVectorRegister,
                      StubOperandAccess::kWrite, 1,
                      StubFragmentKind::kScalar,
                      instruction_name == "V_CVT_F32_FP8" ? 32 : 16) &&
                  ContainsDescriptorWaveSize(
                      decoded, StubOperandRole::kDestination,
                      StubOperandSlotKind::kDestination,
                      StubFragmentKind::kScalar, 0) &&
                  HasSlotKind(decoded, StubOperandSlotKind::kDestination),
              "expected routed scalar VOP1 seed to keep scalar descriptor shapes")) {
        return 1;
      }
      if (!Expect(!decoded.uses_paired_operands,
                  "expected routed scalar VOP1 seed to avoid paired-operand flag")) {
        return 1;
      }
    }
  }

  for (std::string_view instruction_name :
       GetStubDecoderRouteInstructions(StubDecoderRoute::kVop3Sdst)) {
    const StubDecodedInstruction decoded = DecodeVop3SdstStub(instruction_name);
    if (!Expect(decoded.status == StubDecodeStatus::kDecodedStub,
                "expected routed VOP3 SDST seed to decode")) {
      return 1;
    }
    if (!Expect(decoded.uses_scale_path && !HasMatrixSlot(decoded) &&
                    !HasMatrixDescriptor(decoded) &&
                    decoded.operand_slots.binding_count == 5 &&
                    decoded.operand_descriptors.descriptor_count == 5 &&
                    CountSlotsWithFragmentKindAndWaveSize(
                        decoded, StubFragmentKind::kScalar, 0) == 5 &&
                    CountDescriptorsWithFragmentKindAndWaveSize(
                        decoded, StubFragmentKind::kScalar, 0) == 5 &&
                    CountOutputSlots(decoded) == 2 &&
                    CountDescriptorsWithAccess(decoded, StubOperandAccess::kRead) == 3 &&
                    CountDescriptorsWithAccess(decoded, StubOperandAccess::kWrite) == 2 &&
                    !decoded.uses_tensor_memory &&
                    AllSlotsExplicit(decoded) &&
                    AllDescriptorsExplicit(decoded) &&
                    AllSlotWaveSizesAre(decoded, 0) &&
                    AllDescriptorWaveSizesAre(decoded, 0),
                "expected routed VOP3 SDST seed to keep non-matrix wave-size semantics")) {
      return 1;
    }
    if (!Expect(HasDescriptorRole(decoded, StubOperandRole::kScale) &&
                    HasDescriptorRole(decoded, StubOperandRole::kDestination) &&
                    ContainsSlot(decoded, StubOperandSlotKind::kDestination,
                                 StubOperandValueClass::kVectorRegister, 0, 2,
                                 true) &&
                    ContainsSlot(decoded,
                                 StubOperandSlotKind::kScalarDestination,
                                 StubOperandValueClass::kScalarRegister, 1, 1,
                                 true) &&
                    ContainsSlot(decoded, StubOperandSlotKind::kSource0,
                                 StubOperandValueClass::kVectorRegister, 2, 2,
                                 false) &&
                    ContainsSlot(decoded, StubOperandSlotKind::kSource1,
                                 StubOperandValueClass::kVectorRegister, 3, 2,
                                 false) &&
                    ContainsSlot(decoded,
                                 StubOperandSlotKind::kScaleSource,
                                 StubOperandValueClass::kVectorRegister, 4, 2,
                                 false) &&
                    ContainsSlotFragment(decoded,
                                         StubOperandSlotKind::kDestination,
                                         StubFragmentKind::kScalar, 1, 1, 1,
                                         64, 1) &&
                    ContainsSlotWaveSize(decoded,
                                         StubOperandSlotKind::kDestination,
                                         StubFragmentKind::kScalar, 0) &&
                    ContainsSlotFragment(decoded,
                                         StubOperandSlotKind::kScalarDestination,
                                         StubFragmentKind::kScalar, 1, 1, 1,
                                         32, 1) &&
                    ContainsSlotWaveSize(decoded,
                                         StubOperandSlotKind::kScalarDestination,
                                         StubFragmentKind::kScalar, 0) &&
                    ContainsSlotFragment(decoded, StubOperandSlotKind::kSource0,
                                         StubFragmentKind::kScalar, 1, 1, 1,
                                         64, 1) &&
                    ContainsSlotWaveSize(decoded, StubOperandSlotKind::kSource0,
                                         StubFragmentKind::kScalar, 0) &&
                    ContainsSlotFragment(decoded, StubOperandSlotKind::kSource1,
                                         StubFragmentKind::kScalar, 1, 1, 1,
                                         64, 1) &&
                    ContainsSlotWaveSize(decoded, StubOperandSlotKind::kSource1,
                                         StubFragmentKind::kScalar, 0) &&
                    ContainsSlotFragment(decoded,
                                         StubOperandSlotKind::kScaleSource,
                                         StubFragmentKind::kScalar, 1, 1, 1,
                                         64, 1) &&
                    ContainsSlotWaveSize(decoded,
                                         StubOperandSlotKind::kScaleSource,
                                         StubFragmentKind::kScalar, 0) &&
                    HasSlotKind(decoded, StubOperandSlotKind::kScalarDestination) &&
                    CountDescriptorsForRole(decoded, StubOperandRole::kScale) == 1 &&
                    CountDescriptorsForRole(decoded, StubOperandRole::kSource0) == 1 &&
                    CountDescriptorsForRole(decoded, StubOperandRole::kSource1) == 1 &&
                    CountDescriptorsForRole(decoded, StubOperandRole::kDestination) == 2 &&
                    CountSlotsOfKind(decoded, StubOperandSlotKind::kDestination) == 1 &&
                    CountSlotsOfKind(decoded, StubOperandSlotKind::kScalarDestination) == 1 &&
                    CountSlotsOfKind(decoded, StubOperandSlotKind::kScaleSource) == 1 &&
                    ContainsDescriptor(decoded, StubOperandRole::kScale,
                                       StubOperandSlotKind::kScaleSource,
                                       StubOperandValueClass::kVectorRegister,
                                       StubOperandAccess::kRead, 2,
                                       StubFragmentKind::kScalar, 64) &&
                    ContainsDescriptorWaveSize(decoded,
                                               StubOperandRole::kScale,
                                               StubOperandSlotKind::kScaleSource,
                                               StubFragmentKind::kScalar, 0) &&
                    ContainsDescriptor(decoded, StubOperandRole::kDestination,
                                       StubOperandSlotKind::kScalarDestination,
                                       StubOperandValueClass::kScalarRegister,
                                       StubOperandAccess::kWrite, 1,
                                       StubFragmentKind::kScalar, 32) &&
                    ContainsDescriptorWaveSize(
                        decoded, StubOperandRole::kDestination,
                        StubOperandSlotKind::kScalarDestination,
                        StubFragmentKind::kScalar, 0) &&
                    ContainsDescriptor(decoded, StubOperandRole::kDestination,
                                       StubOperandSlotKind::kDestination,
                                       StubOperandValueClass::kVectorRegister,
                                       StubOperandAccess::kWrite, 2,
                                       StubFragmentKind::kScalar, 64) &&
                    ContainsDescriptorWaveSize(
                        decoded, StubOperandRole::kDestination,
                        StubOperandSlotKind::kDestination,
                        StubFragmentKind::kScalar, 0),
                "expected routed VOP3 SDST seed to keep scale/destination descriptors")) {
      return 1;
    }
  }

  for (std::string_view instruction_name :
       GetStubDecoderRouteInstructions(StubDecoderRoute::kVop3p)) {
    if (instruction_name.rfind("V_WMMA_LD_SCALE", 0) != 0) {
      continue;
    }
    const StubDecodedInstruction decoded = DecodeVop3pStub(instruction_name);
    if (!Expect(decoded.status == StubDecodeStatus::kDecodedStub,
                "expected routed paired-scale seed to decode")) {
      return 1;
    }
    if (!Expect(decoded.uses_scale_path && decoded.uses_paired_operands &&
                    !HasMatrixSlot(decoded) && !HasMatrixDescriptor(decoded) &&
                    !decoded.uses_tensor_memory &&
                    decoded.operand_slots.binding_count == 4 &&
                    decoded.operand_descriptors.descriptor_count == 4 &&
                    CountSlotsWithFragmentKindAndWaveSize(
                        decoded, StubFragmentKind::kVector, 0) == 2 &&
                    CountSlotsWithFragmentKindAndWaveSize(
                        decoded, StubFragmentKind::kScalar, 0) == 2 &&
                    CountDescriptorsWithFragmentKindAndWaveSize(
                        decoded, StubFragmentKind::kVector, 0) == 2 &&
                    CountDescriptorsWithFragmentKindAndWaveSize(
                        decoded, StubFragmentKind::kScalar, 0) == 2 &&
                    CountOutputSlots(decoded) == 1 &&
                    CountDescriptorsWithAccess(decoded, StubOperandAccess::kRead) == 3 &&
                    CountDescriptorsWithAccess(decoded, StubOperandAccess::kWrite) == 1 &&
                    AllSlotsExplicit(decoded) &&
                    AllDescriptorsExplicit(decoded) &&
                    AllSlotWaveSizesAre(decoded, 0) &&
                    AllDescriptorWaveSizesAre(decoded, 0),
                "expected paired-scale helper to keep non-matrix wave-size semantics")) {
      return 1;
    }
    if (!Expect(HasDescriptorRole(decoded, StubOperandRole::kSource0) &&
                    HasDescriptorRole(decoded, StubOperandRole::kScale) &&
                    HasDescriptorRole(decoded, StubOperandRole::kPairedScale) &&
                    HasDescriptorRole(decoded, StubOperandRole::kDestination) &&
                    ContainsSlot(decoded, StubOperandSlotKind::kDestination,
                                 StubOperandValueClass::kVectorRegister, 0, 1,
                                 true) &&
                    ContainsSlot(decoded, StubOperandSlotKind::kSource0,
                                 StubOperandValueClass::kVectorRegister, 1, 1,
                                 false) &&
                    ContainsSlot(decoded,
                                 StubOperandSlotKind::kScaleSource,
                                 StubOperandValueClass::kScalarRegister, 2, 1,
                                 false) &&
                    ContainsSlot(decoded,
                                 StubOperandSlotKind::kPairedScaleSource,
                                 StubOperandValueClass::kScalarRegister, 3, 1,
                                 false) &&
                    ContainsSlotFragment(
                        decoded, StubOperandSlotKind::kSource0,
                        StubFragmentKind::kVector, 1, 1, 1,
                        instruction_name == "V_WMMA_LD_SCALE16_PAIRED_B64" ? 64
                                                                           : 32,
                        1) &&
                    ContainsSlotWaveSize(decoded, StubOperandSlotKind::kSource0,
                                         StubFragmentKind::kVector, 0) &&
                    ContainsSlotFragment(
                        decoded, StubOperandSlotKind::kDestination,
                        StubFragmentKind::kVector, 1, 1, 1,
                        instruction_name == "V_WMMA_LD_SCALE16_PAIRED_B64" ? 64
                                                                           : 32,
                        1) &&
                    ContainsSlotWaveSize(decoded,
                                         StubOperandSlotKind::kDestination,
                                         StubFragmentKind::kVector, 0) &&
                    ContainsSlotFragment(decoded,
                                         StubOperandSlotKind::kScaleSource,
                                         StubFragmentKind::kScalar, 1, 1, 1,
                                         32, 1) &&
                    ContainsSlotWaveSize(decoded,
                                         StubOperandSlotKind::kScaleSource,
                                         StubFragmentKind::kScalar, 0) &&
                    ContainsSlotFragment(
                        decoded, StubOperandSlotKind::kPairedScaleSource,
                        StubFragmentKind::kScalar, 1, 1, 1, 32, 1) &&
                    ContainsSlotWaveSize(
                        decoded, StubOperandSlotKind::kPairedScaleSource,
                        StubFragmentKind::kScalar, 0) &&
                    CountDescriptorsForRole(decoded, StubOperandRole::kSource0) == 1 &&
                    CountDescriptorsForRole(decoded, StubOperandRole::kScale) == 1 &&
                    CountDescriptorsForRole(decoded, StubOperandRole::kPairedScale) == 1 &&
                    CountDescriptorsForRole(decoded, StubOperandRole::kDestination) == 1 &&
                    CountSlotsOfKind(decoded, StubOperandSlotKind::kSource0) == 1 &&
                    CountSlotsOfKind(decoded, StubOperandSlotKind::kScaleSource) == 1 &&
                    CountSlotsOfKind(decoded, StubOperandSlotKind::kPairedScaleSource) == 1 &&
                    CountSlotsOfKind(decoded, StubOperandSlotKind::kDestination) == 1 &&
                    ContainsDescriptor(decoded, StubOperandRole::kScale,
                                       StubOperandSlotKind::kScaleSource,
                                       StubOperandValueClass::kScalarRegister,
                                       StubOperandAccess::kRead, 1,
                                       StubFragmentKind::kScalar, 32) &&
                    ContainsDescriptorWaveSize(decoded,
                                               StubOperandRole::kScale,
                                               StubOperandSlotKind::kScaleSource,
                                               StubFragmentKind::kScalar, 0) &&
                    ContainsDescriptor(decoded, StubOperandRole::kPairedScale,
                                       StubOperandSlotKind::kPairedScaleSource,
                                       StubOperandValueClass::kScalarRegister,
                                       StubOperandAccess::kRead, 1,
                                       StubFragmentKind::kScalar, 32) &&
                    ContainsDescriptorWaveSize(
                        decoded, StubOperandRole::kPairedScale,
                        StubOperandSlotKind::kPairedScaleSource,
                        StubFragmentKind::kScalar, 0) &&
                    ContainsDescriptor(
                        decoded, StubOperandRole::kDestination,
                        StubOperandSlotKind::kDestination,
                        StubOperandValueClass::kVectorRegister,
                        StubOperandAccess::kWrite, 1,
                        StubFragmentKind::kVector,
                        instruction_name == "V_WMMA_LD_SCALE16_PAIRED_B64" ? 64
                                                                           : 32) &&
                    ContainsDescriptorWaveSize(
                        decoded, StubOperandRole::kDestination,
                        StubOperandSlotKind::kDestination,
                        StubFragmentKind::kVector, 0),
                "expected paired-scale helper descriptor roles")) {
      return 1;
    }
  }

  const StubDecodedInstruction wrong_route =
      DecodeVop1Stub("V_PK_ADD_BF16");
  if (!Expect(wrong_route.status == StubDecodeStatus::kUnsupportedRoute,
              "expected wrong stub entrypoint to reject VOP3P op")) {
    return 1;
  }

  const StubDecodedInstruction unsupported =
      DecodeStubInstruction("V_CVT_PK_FP8_F32");
  if (!Expect(unsupported.status == StubDecodeStatus::kUnsupportedRoute,
              "expected first-pass unsupported route for VOP3-only seed")) {
    return 1;
  }
  if (!Expect(unsupported.entrypoint_name == "DecodeUnsupportedStub",
              "expected unsupported route to surface unsupported entrypoint")) {
    return 1;
  }

  const StubDecodedInstruction unknown =
      DecodeStubInstruction("NO_SUCH_GFX1250_OPCODE");
  if (!Expect(unknown.status == StubDecodeStatus::kUnknownInstruction,
              "expected unknown instruction status for missing opcode")) {
    return 1;
  }
  if (!Expect(unknown.opcode_shape == StubOpcodeShape::kUnknown,
              "expected unknown opcode shape for missing opcode")) {
    return 1;
  }

  const StubDecoderEntrypointManifest* vop3p_manifest =
      FindStubDecoderEntrypointManifest(StubDecoderRoute::kVop3p);
  if (!Expect(vop3p_manifest != nullptr,
              "expected VOP3P entrypoint manifest")) {
    return 1;
  }
  if (!Expect(vop3p_manifest->entrypoint_name == "DecodeVop3pStub",
              "expected VOP3P entrypoint manifest name")) {
    return 1;
  }

  std::size_t total_manifest_instructions = 0;
  for (const StubDecoderEntrypointManifest& manifest :
       GetStubDecoderEntrypointManifests()) {
    total_manifest_instructions += manifest.instruction_count;
  }
  if (!Expect(total_manifest_instructions ==
                  GetStubDecoderRouteInfos().size(),
              "expected entrypoint manifests to cover all routed seeds")) {
    return 1;
  }

  const auto* route_info = FindStubDecoderRouteInfo("V_WMMA_F32_16X16X4_F32_w32");
  if (!Expect(route_info != nullptr, "expected WMMA route info lookup")) {
    return 1;
  }
  const StubDecodedInstruction via_route_info = DecodeStubInstruction(*route_info);
  if (!Expect(via_route_info.entrypoint_name == "DecodeVop3pStub",
              "expected WMMA route info to dispatch through VOP3P stub")) {
    return 1;
  }
  if (!Expect(GetStubOpcodeShapeName(via_route_info.opcode_shape) == "kWmmaCore",
              "expected opcode-shape name helper to match WMMA route")) {
    return 1;
  }
  if (!Expect(GetStubExecutionDomainName(via_route_info.execution_domain) ==
                  "kMatrix",
              "expected execution-domain helper to match WMMA route")) {
    return 1;
  }
  if (!Expect(GetStubOperandLayoutName(via_route_info.operand_layout.layout_kind) ==
                  "kWmmaF32_16x16x4_F32W32",
              "expected operand-layout helper to match WMMA route")) {
    return 1;
  }
  if (!Expect(GetStubOperandLayoutName(
                  wmma_bf16f32_generic.operand_layout.layout_kind) ==
                  "kWmmaCoreGeneric" &&
                  GetStubOperandLayoutName(
                      swmmac_bf16f32_generic.operand_layout.layout_kind) ==
                      "kSwmmacCoreGeneric" &&
                  GetStubOperandLayoutName(
                      wmma_scale_f4_generic.operand_layout.layout_kind) ==
                      "kWmmaScaleGeneric",
              "expected operand-layout helper to expose generic routed names")) {
    return 1;
  }
  if (!Expect(GetStubOperandRoleName(via_route_info.operand_roles.bindings[0].role) ==
                  "kSource0",
              "expected operand-role helper to match WMMA route")) {
    return 1;
  }
  if (!Expect(GetStubOperandSlotKindName(via_route_info.operand_slots.bindings[0].slot_kind) ==
                  "kDestination",
              "expected operand-slot helper to match WMMA route")) {
    return 1;
  }
  if (!Expect(GetStubOperandValueClassName(via_route_info.operand_slots.bindings[0].value_class) ==
                  "kMatrixFragment",
              "expected operand-value-class helper to match WMMA route")) {
    return 1;
  }

  return 0;
}
