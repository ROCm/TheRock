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

}  // namespace

int main() {
  if (!Expect(GetStubDecoderEntrypointManifests().size() == 4,
              "expected four stub decoder entrypoint manifests")) {
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
  if (!Expect(ContainsSlotFragment(wmma128_fp8,
                                   StubOperandSlotKind::kDestination,
                                   StubFragmentKind::kMatrix, 16, 16, 128, 32,
                                   0),
              "expected WMMA 128 FP8 destination fragment shape")) {
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

  const StubDecodedInstruction wmma_scale16 =
      DecodeVop3pStub("V_WMMA_SCALE16_F32_16X16X128_F8F6F4");
  if (!Expect(wmma_scale16.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaScale16F32_16x16x128_F8F6F4,
              "expected WMMA scale16 operand layout")) {
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

  const StubDecodedInstruction paired_scale16 =
      DecodeVop3pStub("V_WMMA_LD_SCALE16_PAIRED_B64");
  if (!Expect(paired_scale16.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kWmmaLdScale16PairedB64,
              "expected paired WMMA scale16 operand layout")) {
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
                  tensor, StubOperandRole::kLdsDestination,
                  StubOperandSlotKind::kLdsDestination,
                  StubOperandValueClass::kLdsAddress, StubOperandAccess::kWrite,
                  1, StubFragmentKind::kAddress, 32),
              "expected tensor-load LDS destination descriptor")) {
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

  const StubDecodedInstruction packed_bf8 =
      DecodeVop1Stub("V_CVT_PK_F16_BF8");
  if (!Expect(packed_bf8.operand_layout.layout_kind ==
                  StubOperandLayoutKind::kCvtPkF16Bf8,
              "expected explicit operand layout for V_CVT_PK_F16_BF8")) {
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
