#include <iostream>
#include <initializer_list>
#include <string_view>

#include "lib/sim/isa/gfx1250/decoder_seed_catalog.h"
#include "lib/sim/isa/gfx1250/stub_decoder.h"

namespace {

using mirage::sim::isa::gfx1250::DecodeMimgTensorStub;
using mirage::sim::isa::gfx1250::DecodeSeedHint;
using mirage::sim::isa::gfx1250::DecodeStubInstruction;
using mirage::sim::isa::gfx1250::DecodeVop1Stub;
using mirage::sim::isa::gfx1250::DecodeVop3SdstStub;
using mirage::sim::isa::gfx1250::DecodeVop3pStub;
using mirage::sim::isa::gfx1250::DecoderSeedInfo;
using mirage::sim::isa::gfx1250::FindStubDecoderEntrypointManifest;
using mirage::sim::isa::gfx1250::FindStubDecoderRouteInfo;
using mirage::sim::isa::gfx1250::FindStubDecoderRouteManifest;
using mirage::sim::isa::gfx1250::GetDecoderSeedInfos;
using mirage::sim::isa::gfx1250::GetStubDecoderEntrypointManifests;
using mirage::sim::isa::gfx1250::GetStubDecoderRouteInstructions;
using mirage::sim::isa::gfx1250::GetStubDecoderRouteManifests;
using mirage::sim::isa::gfx1250::GetStubExecutionDomainName;
using mirage::sim::isa::gfx1250::GetStubOperandLayoutName;
using mirage::sim::isa::gfx1250::GetStubOperandRoleName;
using mirage::sim::isa::gfx1250::GetStubOperandSlotKindName;
using mirage::sim::isa::gfx1250::GetStubOperandValueClassName;
using mirage::sim::isa::gfx1250::GetStubOpcodeShapeName;
using mirage::sim::isa::gfx1250::GetStubDecoderRouteInfos;
using mirage::sim::isa::gfx1250::SelectStubDecoderRoute;
using mirage::sim::isa::gfx1250::StubDecodedInstruction;
using mirage::sim::isa::gfx1250::StubDecodeStatus;
using mirage::sim::isa::gfx1250::StubDecoderEntrypointManifest;
using mirage::sim::isa::gfx1250::StubDecoderRoute;
using mirage::sim::isa::gfx1250::StubDecoderRouteInfo;
using mirage::sim::isa::gfx1250::StubDecoderRouteManifest;
using mirage::sim::isa::gfx1250::StubExecutionDomain;
using mirage::sim::isa::gfx1250::StubOperandAccess;
using mirage::sim::isa::gfx1250::StubFragmentKind;
using mirage::sim::isa::gfx1250::StubOperandLayoutKind;
using mirage::sim::isa::gfx1250::StubOperandRole;
using mirage::sim::isa::gfx1250::StubOperandSlotKind;
using mirage::sim::isa::gfx1250::StubOperandValueClass;
using mirage::sim::isa::gfx1250::StubOpcodeShape;

struct ShapeExtents {
  std::uint16_t rows = 0xffff;
  std::uint16_t columns = 0xffff;
  std::uint16_t depth = 0xffff;
  bool valid = false;
};

struct ExpectedRoleBinding {
  StubOperandRole role;
  std::uint32_t count;
  bool is_output;
  bool is_implicit;
};

struct ExpectedSlotBinding {
  StubOperandSlotKind slot_kind;
  StubOperandValueClass value_class;
  std::uint32_t logical_operand_index;
  std::uint32_t component_count;
  bool is_output;
  bool is_implicit;
};

struct ExpectedDescriptorBinding {
  StubOperandRole role;
  StubOperandSlotKind slot_kind;
  StubOperandValueClass value_class;
  StubOperandAccess access;
  std::uint8_t component_count;
  bool is_implicit;
};

struct ExpectedRouteMetadata {
  StubDecoderRoute route;
  std::string_view route_name;
  std::string_view entrypoint_name;
  std::uint32_t route_priority;
};

struct ExpectedLayout {
  StubOperandLayoutKind layout_kind = StubOperandLayoutKind::kUnknown;
  std::uint32_t source_count = 0;
  std::uint32_t destination_count = 0;
  std::uint32_t accumulator_source_count = 0;
  bool has_scale_operand = false;
  bool has_paired_scale_operand = false;
  bool has_tensor_descriptor = false;
  bool touches_lds = false;
  bool is_store = false;
};

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

std::uint32_t CountRoleBindings(const StubDecodedInstruction& instruction,
                                StubOperandRole role) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_roles.binding_count; ++i) {
    if (instruction.operand_roles.bindings[i].role == role) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountRoleBindingsWithCount(const StubDecodedInstruction& instruction,
                                         StubOperandRole role,
                                         std::uint32_t binding_count_value) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_roles.binding_count; ++i) {
    const auto& binding = instruction.operand_roles.bindings[i];
    if (binding.role == role && binding.count == binding_count_value) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountRoleBindingsWithOutputFlag(
    const StubDecodedInstruction& instruction,
    StubOperandRole role,
    bool is_output) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_roles.binding_count; ++i) {
    const auto& binding = instruction.operand_roles.bindings[i];
    if (binding.role == role && binding.is_output == is_output) {
      ++count;
    }
  }
  return count;
}

bool AllRoleBindingsExplicit(const StubDecodedInstruction& instruction) {
  for (std::uint32_t i = 0; i < instruction.operand_roles.binding_count; ++i) {
    if (instruction.operand_roles.bindings[i].is_implicit) {
      return false;
    }
  }
  return true;
}

bool MatchesRoleBindingSequence(
    const StubDecodedInstruction& instruction,
    std::initializer_list<ExpectedRoleBinding> expected_bindings) {
  if (instruction.operand_roles.binding_count != expected_bindings.size()) {
    return false;
  }
  std::uint32_t index = 0;
  for (const ExpectedRoleBinding& expected : expected_bindings) {
    const auto& binding = instruction.operand_roles.bindings[index++];
    if (binding.role != expected.role || binding.count != expected.count ||
        binding.is_output != expected.is_output ||
        binding.is_implicit != expected.is_implicit) {
      return false;
    }
  }
  return true;
}

bool MatchesSlotBindingSequence(
    const StubDecodedInstruction& instruction,
    std::initializer_list<ExpectedSlotBinding> expected_bindings) {
  if (instruction.operand_slots.binding_count != expected_bindings.size()) {
    return false;
  }
  std::uint32_t index = 0;
  for (const ExpectedSlotBinding& expected : expected_bindings) {
    const auto& binding = instruction.operand_slots.bindings[index++];
    if (binding.slot_kind != expected.slot_kind ||
        binding.value_class != expected.value_class ||
        binding.logical_operand_index != expected.logical_operand_index ||
        binding.component_count != expected.component_count ||
        binding.is_output != expected.is_output ||
        binding.is_implicit != expected.is_implicit) {
      return false;
    }
  }
  return true;
}

bool MatchesDescriptorSequence(
    const StubDecodedInstruction& instruction,
    std::initializer_list<ExpectedDescriptorBinding> expected_bindings) {
  if (instruction.operand_descriptors.descriptor_count !=
      expected_bindings.size()) {
    return false;
  }
  std::uint32_t index = 0;
  for (const ExpectedDescriptorBinding& expected : expected_bindings) {
    const auto& descriptor = instruction.operand_descriptors.descriptors[index++];
    if (descriptor.role != expected.role ||
        descriptor.slot_kind != expected.slot_kind ||
        descriptor.value_class != expected.value_class ||
        descriptor.access != expected.access ||
        descriptor.component_count != expected.component_count ||
        descriptor.is_implicit != expected.is_implicit) {
      return false;
    }
  }
  return true;
}

bool MatchesRouteMetadata(const StubDecodedInstruction& instruction,
                          const ExpectedRouteMetadata& expected) {
  return instruction.route == expected.route &&
         instruction.route_name == expected.route_name &&
         instruction.entrypoint_name == expected.entrypoint_name &&
         instruction.route_priority == expected.route_priority;
}

bool MatchesLayout(const StubDecodedInstruction& instruction,
                   const ExpectedLayout& expected) {
  return instruction.operand_layout.layout_kind == expected.layout_kind &&
         instruction.operand_layout.source_count == expected.source_count &&
         instruction.operand_layout.destination_count ==
             expected.destination_count &&
         instruction.operand_layout.accumulator_source_count ==
             expected.accumulator_source_count &&
         instruction.operand_layout.has_scale_operand ==
             expected.has_scale_operand &&
         instruction.operand_layout.has_paired_scale_operand ==
             expected.has_paired_scale_operand &&
         instruction.operand_layout.has_tensor_descriptor ==
             expected.has_tensor_descriptor &&
         instruction.operand_layout.touches_lds == expected.touches_lds &&
         instruction.operand_layout.is_store == expected.is_store;
}

std::uint32_t CountDescriptorsForRole(const StubDecodedInstruction& instruction,
                                      StubOperandRole role);
std::uint32_t CountSlotsOfKind(const StubDecodedInstruction& instruction,
                               StubOperandSlotKind slot_kind);
std::uint32_t CountDescriptorsWithAccess(
    const StubDecodedInstruction& instruction,
    StubOperandAccess access);
std::uint32_t CountOutputSlots(const StubDecodedInstruction& instruction);

bool MatchesLayoutToRecordInvariants(const StubDecodedInstruction& instruction) {
  const std::uint32_t paired_scale_count =
      instruction.operand_layout.has_paired_scale_operand ? 1u : 0u;
  const std::uint32_t lds_count =
      instruction.operand_layout.touches_lds ? 1u : 0u;
  const std::uint32_t expected_record_count =
      instruction.operand_layout.source_count +
      instruction.operand_layout.destination_count +
      instruction.operand_layout.accumulator_source_count +
      paired_scale_count + lds_count;
  const std::uint32_t expected_output_count =
      instruction.operand_layout.destination_count +
      (instruction.operand_layout.touches_lds &&
               !instruction.operand_layout.is_store
           ? 1u
           : 0u);
  const std::uint32_t expected_read_descriptor_count =
      instruction.operand_layout.source_count +
      instruction.operand_layout.accumulator_source_count +
      paired_scale_count +
      (instruction.operand_layout.touches_lds && instruction.operand_layout.is_store
           ? 1u
           : 0u);
  const std::uint32_t expected_write_descriptor_count = expected_output_count;

  return instruction.operand_slots.binding_count == expected_record_count &&
         instruction.operand_descriptors.descriptor_count == expected_record_count &&
         CountOutputSlots(instruction) == expected_output_count &&
         CountDescriptorsWithAccess(instruction, StubOperandAccess::kRead) ==
             expected_read_descriptor_count &&
         CountDescriptorsWithAccess(instruction, StubOperandAccess::kWrite) ==
             expected_write_descriptor_count &&
         CountRoleBindings(instruction, StubOperandRole::kAccumulator) ==
             instruction.operand_layout.accumulator_source_count &&
         CountSlotsOfKind(instruction, StubOperandSlotKind::kAccumulatorSource) ==
             instruction.operand_layout.accumulator_source_count &&
         CountDescriptorsForRole(instruction, StubOperandRole::kAccumulator) ==
             instruction.operand_layout.accumulator_source_count &&
         CountRoleBindings(instruction, StubOperandRole::kScale) ==
             (instruction.operand_layout.has_scale_operand ? 1u : 0u) &&
         CountSlotsOfKind(instruction, StubOperandSlotKind::kScaleSource) ==
             (instruction.operand_layout.has_scale_operand ? 1u : 0u) &&
         CountDescriptorsForRole(instruction, StubOperandRole::kScale) ==
             (instruction.operand_layout.has_scale_operand ? 1u : 0u) &&
         CountRoleBindings(instruction, StubOperandRole::kPairedScale) ==
             paired_scale_count &&
         CountSlotsOfKind(instruction, StubOperandSlotKind::kPairedScaleSource) ==
             paired_scale_count &&
         CountDescriptorsForRole(instruction, StubOperandRole::kPairedScale) ==
             paired_scale_count &&
         CountRoleBindings(instruction, StubOperandRole::kTensorDescriptor) ==
             (instruction.operand_layout.has_tensor_descriptor ? 1u : 0u) &&
         CountSlotsOfKind(instruction,
                          StubOperandSlotKind::kTensorDescriptorSource) ==
             (instruction.operand_layout.has_tensor_descriptor ? 1u : 0u) &&
         CountDescriptorsForRole(instruction, StubOperandRole::kTensorDescriptor) ==
             (instruction.operand_layout.has_tensor_descriptor ? 1u : 0u) &&
         CountRoleBindings(instruction, StubOperandRole::kTensorCoordinate) ==
             (instruction.operand_layout.has_tensor_descriptor ? 1u : 0u) &&
         CountSlotsOfKind(instruction,
                          StubOperandSlotKind::kTensorCoordinateSource) ==
             (instruction.operand_layout.has_tensor_descriptor ? 1u : 0u) &&
         CountDescriptorsForRole(instruction, StubOperandRole::kTensorCoordinate) ==
             (instruction.operand_layout.has_tensor_descriptor ? 1u : 0u) &&
         CountRoleBindings(instruction, StubOperandRole::kLdsDestination) ==
             (instruction.operand_layout.touches_lds &&
                      !instruction.operand_layout.is_store
                  ? 1u
                  : 0u) &&
         CountSlotsOfKind(instruction, StubOperandSlotKind::kLdsDestination) ==
             (instruction.operand_layout.touches_lds &&
                      !instruction.operand_layout.is_store
                  ? 1u
                  : 0u) &&
         CountDescriptorsForRole(instruction, StubOperandRole::kLdsDestination) ==
             (instruction.operand_layout.touches_lds &&
                      !instruction.operand_layout.is_store
                  ? 1u
                  : 0u) &&
         CountRoleBindings(instruction, StubOperandRole::kLdsSource) ==
             (instruction.operand_layout.touches_lds &&
                      instruction.operand_layout.is_store
                  ? 1u
                  : 0u) &&
         CountSlotsOfKind(instruction, StubOperandSlotKind::kLdsSource) ==
             (instruction.operand_layout.touches_lds &&
                      instruction.operand_layout.is_store
                  ? 1u
                  : 0u) &&
         CountDescriptorsForRole(instruction, StubOperandRole::kLdsSource) ==
             (instruction.operand_layout.touches_lds &&
                      instruction.operand_layout.is_store
                  ? 1u
                  : 0u);
}

bool MatchesRouteInfoPayload(const StubDecodedInstruction& instruction,
                             const StubDecoderRouteInfo& route_info) {
  return instruction.instruction_name == route_info.instruction_name &&
         instruction.route == route_info.route &&
         instruction.route_name == route_info.route_name &&
         instruction.route_priority == route_info.route_priority &&
         instruction.rdna4_encoding_name == route_info.rdna4_encoding_name &&
         instruction.rdna4_opcode == route_info.rdna4_opcode &&
         instruction.rdna4_operand_count == route_info.rdna4_operand_count &&
         instruction.appears_in_rdna4_xml == route_info.appears_in_rdna4_xml &&
         instruction.is_target_specific == route_info.is_target_specific;
}

bool MatchesTopLevelFlags(const StubDecodedInstruction& instruction,
                          bool uses_accumulator,
                          bool uses_tensor_memory,
                          bool uses_scale_path,
                          bool uses_paired_operands) {
  return instruction.uses_accumulator == uses_accumulator &&
         instruction.uses_tensor_memory == uses_tensor_memory &&
         instruction.uses_scale_path == uses_scale_path &&
         instruction.uses_paired_operands == uses_paired_operands;
}

bool MatchesFragmentShape(
    const mirage::sim::isa::gfx1250::StubFragmentShape& lhs,
    const mirage::sim::isa::gfx1250::StubFragmentShape& rhs) {
  return lhs.kind == rhs.kind && lhs.rows == rhs.rows &&
         lhs.columns == rhs.columns && lhs.depth == rhs.depth &&
         lhs.element_bit_width == rhs.element_bit_width &&
         lhs.packed_elements == rhs.packed_elements &&
         lhs.wave_size == rhs.wave_size;
}

bool MatchesOperandRoleRecord(
    const mirage::sim::isa::gfx1250::StubOperandRoleRecord& lhs,
    const mirage::sim::isa::gfx1250::StubOperandRoleRecord& rhs) {
  if (lhs.binding_count != rhs.binding_count) {
    return false;
  }
  for (std::uint32_t i = 0; i < lhs.binding_count; ++i) {
    const auto& lhs_binding = lhs.bindings[i];
    const auto& rhs_binding = rhs.bindings[i];
    if (lhs_binding.role != rhs_binding.role ||
        lhs_binding.count != rhs_binding.count ||
        lhs_binding.is_output != rhs_binding.is_output ||
        lhs_binding.is_implicit != rhs_binding.is_implicit) {
      return false;
    }
  }
  return true;
}

bool MatchesOperandSlotRecord(
    const mirage::sim::isa::gfx1250::StubOperandSlotRecord& lhs,
    const mirage::sim::isa::gfx1250::StubOperandSlotRecord& rhs) {
  if (lhs.binding_count != rhs.binding_count) {
    return false;
  }
  for (std::uint32_t i = 0; i < lhs.binding_count; ++i) {
    const auto& lhs_binding = lhs.bindings[i];
    const auto& rhs_binding = rhs.bindings[i];
    if (lhs_binding.slot_kind != rhs_binding.slot_kind ||
        lhs_binding.value_class != rhs_binding.value_class ||
        lhs_binding.logical_operand_index != rhs_binding.logical_operand_index ||
        lhs_binding.component_count != rhs_binding.component_count ||
        lhs_binding.is_output != rhs_binding.is_output ||
        lhs_binding.is_implicit != rhs_binding.is_implicit ||
        !MatchesFragmentShape(lhs_binding.fragment_shape,
                              rhs_binding.fragment_shape)) {
      return false;
    }
  }
  return true;
}

bool MatchesOperandDescriptorRecord(
    const mirage::sim::isa::gfx1250::StubOperandDescriptorRecord& lhs,
    const mirage::sim::isa::gfx1250::StubOperandDescriptorRecord& rhs) {
  if (lhs.descriptor_count != rhs.descriptor_count) {
    return false;
  }
  for (std::uint32_t i = 0; i < lhs.descriptor_count; ++i) {
    const auto& lhs_descriptor = lhs.descriptors[i];
    const auto& rhs_descriptor = rhs.descriptors[i];
    if (lhs_descriptor.role != rhs_descriptor.role ||
        lhs_descriptor.slot_kind != rhs_descriptor.slot_kind ||
        lhs_descriptor.value_class != rhs_descriptor.value_class ||
        lhs_descriptor.access != rhs_descriptor.access ||
        lhs_descriptor.component_count != rhs_descriptor.component_count ||
        lhs_descriptor.is_implicit != rhs_descriptor.is_implicit ||
        !MatchesFragmentShape(lhs_descriptor.fragment_shape,
                              rhs_descriptor.fragment_shape)) {
      return false;
    }
  }
  return true;
}

bool MatchesDecodedInstruction(const StubDecodedInstruction& lhs,
                               const StubDecodedInstruction& rhs) {
  return lhs.instruction_name == rhs.instruction_name &&
         lhs.status == rhs.status && lhs.route == rhs.route &&
         lhs.route_name == rhs.route_name &&
         lhs.entrypoint_name == rhs.entrypoint_name &&
         lhs.route_priority == rhs.route_priority &&
         lhs.rdna4_encoding_name == rhs.rdna4_encoding_name &&
         lhs.rdna4_opcode == rhs.rdna4_opcode &&
         lhs.rdna4_operand_count == rhs.rdna4_operand_count &&
         lhs.appears_in_rdna4_xml == rhs.appears_in_rdna4_xml &&
         lhs.is_target_specific == rhs.is_target_specific &&
         lhs.opcode_shape == rhs.opcode_shape &&
         lhs.execution_domain == rhs.execution_domain &&
         lhs.uses_accumulator == rhs.uses_accumulator &&
         lhs.uses_tensor_memory == rhs.uses_tensor_memory &&
         lhs.uses_scale_path == rhs.uses_scale_path &&
         lhs.uses_paired_operands == rhs.uses_paired_operands &&
         lhs.operand_layout.layout_kind == rhs.operand_layout.layout_kind &&
         lhs.operand_layout.source_count == rhs.operand_layout.source_count &&
         lhs.operand_layout.destination_count ==
             rhs.operand_layout.destination_count &&
         lhs.operand_layout.accumulator_source_count ==
             rhs.operand_layout.accumulator_source_count &&
         lhs.operand_layout.has_scale_operand ==
             rhs.operand_layout.has_scale_operand &&
         lhs.operand_layout.has_paired_scale_operand ==
             rhs.operand_layout.has_paired_scale_operand &&
         lhs.operand_layout.has_tensor_descriptor ==
             rhs.operand_layout.has_tensor_descriptor &&
         lhs.operand_layout.touches_lds == rhs.operand_layout.touches_lds &&
         lhs.operand_layout.is_store == rhs.operand_layout.is_store &&
         MatchesOperandRoleRecord(lhs.operand_roles, rhs.operand_roles) &&
         MatchesOperandSlotRecord(lhs.operand_slots, rhs.operand_slots) &&
         MatchesOperandDescriptorRecord(lhs.operand_descriptors,
                                        rhs.operand_descriptors);
}

bool MatchesDecodedInstructionStructure(const StubDecodedInstruction& lhs,
                                        const StubDecodedInstruction& rhs) {
  return lhs.instruction_name == rhs.instruction_name &&
         lhs.status == rhs.status &&
         lhs.entrypoint_name == rhs.entrypoint_name &&
         lhs.opcode_shape == rhs.opcode_shape &&
         lhs.execution_domain == rhs.execution_domain &&
         lhs.uses_accumulator == rhs.uses_accumulator &&
         lhs.uses_tensor_memory == rhs.uses_tensor_memory &&
         lhs.uses_scale_path == rhs.uses_scale_path &&
         lhs.uses_paired_operands == rhs.uses_paired_operands &&
         lhs.operand_layout.layout_kind == rhs.operand_layout.layout_kind &&
         lhs.operand_layout.source_count == rhs.operand_layout.source_count &&
         lhs.operand_layout.destination_count ==
             rhs.operand_layout.destination_count &&
         lhs.operand_layout.accumulator_source_count ==
             rhs.operand_layout.accumulator_source_count &&
         lhs.operand_layout.has_scale_operand ==
             rhs.operand_layout.has_scale_operand &&
         lhs.operand_layout.has_paired_scale_operand ==
             rhs.operand_layout.has_paired_scale_operand &&
         lhs.operand_layout.has_tensor_descriptor ==
             rhs.operand_layout.has_tensor_descriptor &&
         lhs.operand_layout.touches_lds == rhs.operand_layout.touches_lds &&
         lhs.operand_layout.is_store == rhs.operand_layout.is_store &&
         MatchesOperandRoleRecord(lhs.operand_roles, rhs.operand_roles) &&
         MatchesOperandSlotRecord(lhs.operand_slots, rhs.operand_slots) &&
         MatchesOperandDescriptorRecord(lhs.operand_descriptors,
                                        rhs.operand_descriptors);
}

std::string_view ExpectedOpcodeShapeName(std::string_view instruction_name) {
  if (instruction_name == "V_PK_FMA_BF16") {
    return "kVop3pPackedFma";
  }
  if (instruction_name.rfind("V_PK_", 0) == 0) {
    return "kVop3pPackedBinary";
  }
  if (instruction_name.rfind("V_WMMA_LD_SCALE", 0) == 0) {
    return "kWmmaScalePairedLoad";
  }
  if (instruction_name.rfind("V_WMMA_SCALE", 0) == 0) {
    return "kWmmaScale";
  }
  if (instruction_name.rfind("V_SWMMAC_", 0) == 0) {
    return "kSwmmacCore";
  }
  if (instruction_name.rfind("V_WMMA_", 0) == 0) {
    return "kWmmaCore";
  }
  if (instruction_name == "TENSOR_LOAD_TO_LDS") {
    return "kTensorLoadToLds";
  }
  if (instruction_name == "TENSOR_STORE_FROM_LDS") {
    return "kTensorStoreFromLds";
  }
  if (instruction_name == "V_CVT_F16_FP8" ||
      instruction_name == "V_CVT_F16_BF8") {
    return "kFp8ConvertToF16";
  }
  if (instruction_name == "V_CVT_F32_FP8") {
    return "kFp8ConvertToF32";
  }
  if (instruction_name.rfind("V_CVT_PK_", 0) == 0) {
    return "kFp8PackedConvert";
  }
  if (instruction_name == "V_DIV_SCALE_F64") {
    return "kVop3SdstScale";
  }
  return "kUnknown";
}

std::string_view ExpectedExecutionDomainName(std::string_view instruction_name) {
  if (instruction_name.rfind("V_PK_", 0) == 0) {
    return "kVectorAlu";
  }
  if (instruction_name.rfind("V_WMMA_", 0) == 0 ||
      instruction_name.rfind("V_SWMMAC_", 0) == 0) {
    return "kMatrix";
  }
  if (instruction_name.rfind("TENSOR_", 0) == 0) {
    return "kTensorMemory";
  }
  if (instruction_name.rfind("V_CVT_", 0) == 0) {
    return "kConversion";
  }
  if (instruction_name == "V_DIV_SCALE_F64") {
    return "kScaleAssist";
  }
  return "kUnknown";
}

std::string_view ExpectedOperandLayoutName(std::string_view instruction_name) {
  if (instruction_name == "V_PK_ADD_BF16") {
    return "kPkAddBf16";
  }
  if (instruction_name == "V_PK_FMA_BF16") {
    return "kPkFmaBf16";
  }
  if (instruction_name == "V_PK_MUL_BF16") {
    return "kPkMulBf16";
  }
  if (instruction_name == "V_PK_MIN_NUM_BF16") {
    return "kPkMinNumBf16";
  }
  if (instruction_name == "V_PK_MAX_NUM_BF16") {
    return "kPkMaxNumBf16";
  }
  if (instruction_name == "V_WMMA_F32_16X16X4_F32_w32") {
    return "kWmmaF32_16x16x4_F32W32";
  }
  if (instruction_name == "V_WMMA_F32_16X16X128_FP8_FP8_w32") {
    return "kWmmaF32_16x16x128_Fp8Fp8W32";
  }
  if (instruction_name == "V_WMMA_F16_16X16X128_FP8_FP8_w32") {
    return "kWmmaF16_16x16x128_Fp8Fp8W32";
  }
  if (instruction_name == "V_WMMA_F32_16X16X64_FP8_FP8_w32") {
    return "kWmmaF32_16x16x64_Fp8Fp8W32";
  }
  if (instruction_name == "V_WMMA_SCALE_F32_16X16X128_F8F6F4") {
    return "kWmmaScaleF32_16x16x128_F8F6F4";
  }
  if (instruction_name == "V_WMMA_SCALE16_F32_16X16X128_F8F6F4") {
    return "kWmmaScale16F32_16x16x128_F8F6F4";
  }
  if (instruction_name.rfind("V_WMMA_SCALE", 0) == 0) {
    return "kWmmaScaleGeneric";
  }
  if (instruction_name == "V_WMMA_LD_SCALE_PAIRED_B32") {
    return "kWmmaLdScalePairedB32";
  }
  if (instruction_name == "V_WMMA_LD_SCALE16_PAIRED_B64") {
    return "kWmmaLdScale16PairedB64";
  }
  if (instruction_name == "V_SWMMAC_F32_16X16X128_FP8_FP8_w32") {
    return "kSwmmacF32_16x16x128_Fp8Fp8W32";
  }
  if (instruction_name == "V_SWMMAC_F16_16X16X128_FP8_FP8_w32") {
    return "kSwmmacF16_16x16x128_Fp8Fp8W32";
  }
  if (instruction_name.rfind("V_SWMMAC_", 0) == 0) {
    return "kSwmmacCoreGeneric";
  }
  if (instruction_name.rfind("V_WMMA_", 0) == 0) {
    return "kWmmaCoreGeneric";
  }
  if (instruction_name == "TENSOR_LOAD_TO_LDS") {
    return "kTensorLoadToLds";
  }
  if (instruction_name == "TENSOR_STORE_FROM_LDS") {
    return "kTensorStoreFromLds";
  }
  if (instruction_name == "V_CVT_F16_BF8") {
    return "kCvtF16Bf8";
  }
  if (instruction_name == "V_CVT_F16_FP8") {
    return "kCvtF16Fp8";
  }
  if (instruction_name == "V_CVT_F32_FP8") {
    return "kCvtF32Fp8";
  }
  if (instruction_name == "V_CVT_PK_F16_FP8") {
    return "kCvtPkF16Fp8";
  }
  if (instruction_name == "V_CVT_PK_F16_BF8") {
    return "kCvtPkF16Bf8";
  }
  if (instruction_name == "V_DIV_SCALE_F64") {
    return "kVDivScaleF64";
  }
  return "kUnknown";
}

bool AllRoleHelperNamesKnown(const StubDecodedInstruction& instruction) {
  for (std::uint32_t index = 0; index < instruction.operand_roles.binding_count;
       ++index) {
    if (GetStubOperandRoleName(instruction.operand_roles.bindings[index].role) ==
        "kUnknown") {
      return false;
    }
  }
  for (std::uint32_t index = 0;
       index < instruction.operand_descriptors.descriptor_count; ++index) {
    if (GetStubOperandRoleName(
            instruction.operand_descriptors.descriptors[index].role) ==
        "kUnknown") {
      return false;
    }
  }
  return true;
}

bool AllSlotKindHelperNamesKnown(const StubDecodedInstruction& instruction) {
  for (std::uint32_t index = 0; index < instruction.operand_slots.binding_count;
       ++index) {
    if (GetStubOperandSlotKindName(
            instruction.operand_slots.bindings[index].slot_kind) == "kUnknown") {
      return false;
    }
  }
  for (std::uint32_t index = 0;
       index < instruction.operand_descriptors.descriptor_count; ++index) {
    if (GetStubOperandSlotKindName(
            instruction.operand_descriptors.descriptors[index].slot_kind) ==
        "kUnknown") {
      return false;
    }
  }
  return true;
}

bool AllValueClassHelperNamesKnown(const StubDecodedInstruction& instruction) {
  for (std::uint32_t index = 0; index < instruction.operand_slots.binding_count;
       ++index) {
    if (GetStubOperandValueClassName(
            instruction.operand_slots.bindings[index].value_class) ==
        "kUnknown") {
      return false;
    }
  }
  for (std::uint32_t index = 0;
       index < instruction.operand_descriptors.descriptor_count; ++index) {
    if (GetStubOperandValueClassName(
            instruction.operand_descriptors.descriptors[index].value_class) ==
        "kUnknown") {
      return false;
    }
  }
  return true;
}

StubDecodedInstruction DecodeViaRouteEntrypoint(
    const StubDecoderRouteInfo& route_info) {
  switch (route_info.route) {
    case StubDecoderRoute::kVop3p:
      return DecodeVop3pStub(route_info.instruction_name);
    case StubDecoderRoute::kMimgTensor:
      return DecodeMimgTensorStub(route_info.instruction_name);
    case StubDecoderRoute::kVop1:
      return DecodeVop1Stub(route_info.instruction_name);
    case StubDecoderRoute::kVop3Sdst:
      return DecodeVop3SdstStub(route_info.instruction_name);
    case StubDecoderRoute::kUnsupported:
      break;
  }
  return {};
}

StubDecodedInstruction DecodeViaExplicitRouteEntrypoint(
    StubDecoderRoute route,
    std::string_view instruction_name) {
  switch (route) {
    case StubDecoderRoute::kVop3p:
      return DecodeVop3pStub(instruction_name);
    case StubDecoderRoute::kMimgTensor:
      return DecodeMimgTensorStub(instruction_name);
    case StubDecoderRoute::kVop1:
      return DecodeVop1Stub(instruction_name);
    case StubDecoderRoute::kVop3Sdst:
      return DecodeVop3SdstStub(instruction_name);
    case StubDecoderRoute::kUnsupported:
      break;
  }
  return {};
}

std::string_view SyntheticUnknownInstructionForRoute(StubDecoderRoute route) {
  switch (route) {
    case StubDecoderRoute::kVop3p:
      return "SYNTHETIC_GFX1250_VOP3P_UNKNOWN";
    case StubDecoderRoute::kMimgTensor:
      return "SYNTHETIC_GFX1250_MIMGTENSOR_UNKNOWN";
    case StubDecoderRoute::kVop1:
      return "SYNTHETIC_GFX1250_VOP1_UNKNOWN";
    case StubDecoderRoute::kVop3Sdst:
      return "SYNTHETIC_GFX1250_VOP3SDST_UNKNOWN";
    case StubDecoderRoute::kUnsupported:
      break;
  }
  return "SYNTHETIC_GFX1250_UNSUPPORTED_UNKNOWN";
}

bool IsUnsupportedSeededInstruction(const DecoderSeedInfo& seed) {
  return seed.decode_hint == DecodeSeedHint::kUnknown ||
         seed.decode_hint == DecodeSeedHint::kVop3;
}

DecodeSeedHint AlternateDecodeHintForRoute(StubDecoderRoute route) {
  switch (route) {
    case StubDecoderRoute::kVop3p:
      return DecodeSeedHint::kVop1;
    case StubDecoderRoute::kMimgTensor:
      return DecodeSeedHint::kVop3Sdst;
    case StubDecoderRoute::kVop1:
      return DecodeSeedHint::kMimgTensor;
    case StubDecoderRoute::kVop3Sdst:
      return DecodeSeedHint::kVop3p;
    case StubDecoderRoute::kUnsupported:
      break;
  }
  return DecodeSeedHint::kVop3;
}

bool MatchesUnsupportedRouteDecodeForRoutedSeed(
    const StubDecodedInstruction& decoded,
    const StubDecoderRouteInfo& route_info) {
  return decoded.instruction_name == route_info.instruction_name &&
         decoded.status == StubDecodeStatus::kUnsupportedRoute &&
         decoded.route == route_info.route &&
         decoded.route_name == route_info.route_name &&
         decoded.entrypoint_name == "DecodeUnsupportedStub" &&
         decoded.route_priority == route_info.route_priority &&
         decoded.rdna4_encoding_name == route_info.rdna4_encoding_name &&
         decoded.rdna4_opcode == route_info.rdna4_opcode &&
         decoded.rdna4_operand_count == route_info.rdna4_operand_count &&
         decoded.appears_in_rdna4_xml == route_info.appears_in_rdna4_xml &&
         decoded.is_target_specific == route_info.is_target_specific &&
         decoded.opcode_shape == StubOpcodeShape::kUnknown &&
         decoded.execution_domain == StubExecutionDomain::kUnknown &&
         !decoded.uses_accumulator && !decoded.uses_tensor_memory &&
         !decoded.uses_scale_path && !decoded.uses_paired_operands &&
         decoded.operand_layout.layout_kind == StubOperandLayoutKind::kUnknown &&
         decoded.operand_roles.binding_count == 0 &&
         decoded.operand_slots.binding_count == 0 &&
         decoded.operand_descriptors.descriptor_count == 0;
}

bool MatchesUnsupportedSeedDecode(const StubDecodedInstruction& decoded,
                                  const DecoderSeedInfo& seed) {
  return decoded.instruction_name == seed.instruction_name &&
         decoded.status == StubDecodeStatus::kUnsupportedRoute &&
         decoded.route == StubDecoderRoute::kUnsupported &&
         decoded.route_name == "kUnsupported" &&
         decoded.entrypoint_name == "DecodeUnsupportedStub" &&
         decoded.route_priority == 0 &&
         decoded.rdna4_encoding_name.empty() &&
         decoded.rdna4_opcode == 0 &&
         decoded.rdna4_operand_count == 0 &&
         !decoded.appears_in_rdna4_xml && !decoded.is_target_specific &&
         decoded.opcode_shape == StubOpcodeShape::kUnknown &&
         decoded.execution_domain == StubExecutionDomain::kUnknown &&
         !decoded.uses_accumulator && !decoded.uses_tensor_memory &&
         !decoded.uses_scale_path && !decoded.uses_paired_operands &&
         decoded.operand_layout.layout_kind == StubOperandLayoutKind::kUnknown &&
         decoded.operand_roles.binding_count == 0 &&
         decoded.operand_slots.binding_count == 0 &&
         decoded.operand_descriptors.descriptor_count == 0;
}

bool MatchesUnsupportedInstructionDecode(
    const StubDecodedInstruction& decoded,
    std::string_view instruction_name) {
  return decoded.instruction_name == instruction_name &&
         decoded.status == StubDecodeStatus::kUnsupportedRoute &&
         decoded.route == StubDecoderRoute::kUnsupported &&
         decoded.route_name == "kUnsupported" &&
         decoded.entrypoint_name == "DecodeUnsupportedStub" &&
         decoded.route_priority == 0 &&
         decoded.rdna4_encoding_name.empty() &&
         decoded.rdna4_opcode == 0 &&
         decoded.rdna4_operand_count == 0 &&
         !decoded.appears_in_rdna4_xml && !decoded.is_target_specific &&
         decoded.opcode_shape == StubOpcodeShape::kUnknown &&
         decoded.execution_domain == StubExecutionDomain::kUnknown &&
         !decoded.uses_accumulator && !decoded.uses_tensor_memory &&
         !decoded.uses_scale_path && !decoded.uses_paired_operands &&
         decoded.operand_layout.layout_kind == StubOperandLayoutKind::kUnknown &&
         decoded.operand_roles.binding_count == 0 &&
         decoded.operand_slots.binding_count == 0 &&
         decoded.operand_descriptors.descriptor_count == 0;
}

bool MatchesUnknownDecode(const StubDecodedInstruction& decoded,
                         std::string_view instruction_name) {
  return decoded.instruction_name == instruction_name &&
         decoded.status == StubDecodeStatus::kUnknownInstruction &&
         decoded.route == StubDecoderRoute::kUnsupported &&
         decoded.route_name == "kUnsupported" &&
         decoded.entrypoint_name == "DecodeUnsupportedStub" &&
         decoded.route_priority == 0 &&
         decoded.rdna4_encoding_name.empty() &&
         decoded.rdna4_opcode == 0 &&
         decoded.rdna4_operand_count == 0 &&
         !decoded.appears_in_rdna4_xml && !decoded.is_target_specific &&
         decoded.opcode_shape == StubOpcodeShape::kUnknown &&
         decoded.execution_domain == StubExecutionDomain::kUnknown &&
         !decoded.uses_accumulator && !decoded.uses_tensor_memory &&
         !decoded.uses_scale_path && !decoded.uses_paired_operands &&
         decoded.operand_layout.layout_kind == StubOperandLayoutKind::kUnknown &&
         decoded.operand_roles.binding_count == 0 &&
         decoded.operand_slots.binding_count == 0 &&
         decoded.operand_descriptors.descriptor_count == 0;
}

bool MatchesUnknownHelperSurface(const StubDecodedInstruction& decoded) {
  return GetStubOpcodeShapeName(decoded.opcode_shape) == "kUnknown" &&
         GetStubExecutionDomainName(decoded.execution_domain) == "kUnknown" &&
         GetStubOperandLayoutName(decoded.operand_layout.layout_kind) ==
             "kUnknown" &&
         AllRoleHelperNamesKnown(decoded) && AllSlotKindHelperNamesKnown(decoded) &&
         AllValueClassHelperNamesKnown(decoded);
}

bool MatchesSelectorDecodeStatusParity(std::string_view instruction_name,
                                      StubDecodeStatus expected_status) {
  const StubDecoderRoute selected_route =
      SelectStubDecoderRoute(instruction_name);
  const StubDecoderRouteInfo* route_info =
      FindStubDecoderRouteInfo(instruction_name);
  const StubDecodedInstruction decoded = DecodeStubInstruction(instruction_name);

  switch (expected_status) {
    case StubDecodeStatus::kDecodedStub:
      return selected_route != StubDecoderRoute::kUnsupported &&
             route_info != nullptr && decoded.status == StubDecodeStatus::kDecodedStub &&
             decoded.route == selected_route;
    case StubDecodeStatus::kUnsupportedRoute:
      return selected_route == StubDecoderRoute::kUnsupported &&
             route_info == nullptr &&
             decoded.status == StubDecodeStatus::kUnsupportedRoute;
    case StubDecodeStatus::kUnknownInstruction:
      return selected_route == StubDecoderRoute::kUnsupported &&
             route_info == nullptr &&
             decoded.status == StubDecodeStatus::kUnknownInstruction;
  }
  return false;
}

StubOperandRole ExpectedRoleForSlotKind(StubOperandSlotKind slot_kind) {
  switch (slot_kind) {
    case StubOperandSlotKind::kDestination:
    case StubOperandSlotKind::kScalarDestination:
      return StubOperandRole::kDestination;
    case StubOperandSlotKind::kSource0:
      return StubOperandRole::kSource0;
    case StubOperandSlotKind::kSource1:
      return StubOperandRole::kSource1;
    case StubOperandSlotKind::kSource2:
      return StubOperandRole::kSource2;
    case StubOperandSlotKind::kAccumulatorSource:
      return StubOperandRole::kAccumulator;
    case StubOperandSlotKind::kScaleSource:
      return StubOperandRole::kScale;
    case StubOperandSlotKind::kPairedScaleSource:
      return StubOperandRole::kPairedScale;
    case StubOperandSlotKind::kTensorDescriptorSource:
      return StubOperandRole::kTensorDescriptor;
    case StubOperandSlotKind::kTensorCoordinateSource:
      return StubOperandRole::kTensorCoordinate;
    case StubOperandSlotKind::kLdsDestination:
      return StubOperandRole::kLdsDestination;
    case StubOperandSlotKind::kLdsSource:
      return StubOperandRole::kLdsSource;
    case StubOperandSlotKind::kUnknown:
      break;
  }
  return StubOperandRole::kUnknown;
}

StubOperandAccess ExpectedAccessForSlotKind(StubOperandSlotKind slot_kind) {
  switch (slot_kind) {
    case StubOperandSlotKind::kDestination:
    case StubOperandSlotKind::kScalarDestination:
    case StubOperandSlotKind::kLdsDestination:
      return StubOperandAccess::kWrite;
    case StubOperandSlotKind::kSource0:
    case StubOperandSlotKind::kSource1:
    case StubOperandSlotKind::kSource2:
    case StubOperandSlotKind::kAccumulatorSource:
    case StubOperandSlotKind::kScaleSource:
    case StubOperandSlotKind::kPairedScaleSource:
    case StubOperandSlotKind::kTensorDescriptorSource:
    case StubOperandSlotKind::kTensorCoordinateSource:
    case StubOperandSlotKind::kLdsSource:
      return StubOperandAccess::kRead;
    case StubOperandSlotKind::kUnknown:
      break;
  }
  return StubOperandAccess::kRead;
}

bool MatchesDescriptorToSlotParity(const StubDecodedInstruction& instruction) {
  if (instruction.operand_descriptors.descriptor_count !=
      instruction.operand_slots.binding_count) {
    return false;
  }

  for (std::uint32_t slot_index = 0; slot_index < instruction.operand_slots.binding_count;
       ++slot_index) {
    const auto& slot = instruction.operand_slots.bindings[slot_index];
    std::uint32_t match_count = 0;
    for (std::uint32_t descriptor_index = 0;
         descriptor_index < instruction.operand_descriptors.descriptor_count;
         ++descriptor_index) {
      const auto& descriptor =
          instruction.operand_descriptors.descriptors[descriptor_index];
      if (descriptor.role == ExpectedRoleForSlotKind(slot.slot_kind) &&
          descriptor.slot_kind == slot.slot_kind &&
          descriptor.value_class == slot.value_class &&
          descriptor.access == ExpectedAccessForSlotKind(slot.slot_kind) &&
          descriptor.component_count == slot.component_count &&
          descriptor.is_implicit == slot.is_implicit &&
          MatchesFragmentShape(descriptor.fragment_shape, slot.fragment_shape)) {
        ++match_count;
      }
    }
    if (match_count != 1) {
      return false;
    }
  }

  for (std::uint32_t descriptor_index = 0;
       descriptor_index < instruction.operand_descriptors.descriptor_count;
       ++descriptor_index) {
    const auto& descriptor =
        instruction.operand_descriptors.descriptors[descriptor_index];
    std::uint32_t match_count = 0;
    for (std::uint32_t slot_index = 0; slot_index < instruction.operand_slots.binding_count;
         ++slot_index) {
      const auto& slot = instruction.operand_slots.bindings[slot_index];
      if (descriptor.role == ExpectedRoleForSlotKind(slot.slot_kind) &&
          descriptor.slot_kind == slot.slot_kind &&
          descriptor.value_class == slot.value_class &&
          descriptor.access == ExpectedAccessForSlotKind(slot.slot_kind) &&
          descriptor.component_count == slot.component_count &&
          descriptor.is_implicit == slot.is_implicit &&
          MatchesFragmentShape(descriptor.fragment_shape, slot.fragment_shape)) {
        ++match_count;
      }
    }
    if (match_count != 1) {
      return false;
    }
  }

  return true;
}

std::uint32_t CountRouteInfosForRoute(StubDecoderRoute route) {
  std::uint32_t count = 0;
  for (const StubDecoderRouteInfo& route_info : GetStubDecoderRouteInfos()) {
    if (route_info.route == route) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountRouteInfosForRouteWithXmlFlag(StubDecoderRoute route,
                                                 bool appears_in_rdna4_xml) {
  std::uint32_t count = 0;
  for (const StubDecoderRouteInfo& route_info : GetStubDecoderRouteInfos()) {
    if (route_info.route == route &&
        route_info.appears_in_rdna4_xml == appears_in_rdna4_xml) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountRouteInfosForRouteWithTargetSpecificFlag(
    StubDecoderRoute route,
    bool is_target_specific) {
  std::uint32_t count = 0;
  for (const StubDecoderRouteInfo& route_info : GetStubDecoderRouteInfos()) {
    if (route_info.route == route &&
        route_info.is_target_specific == is_target_specific) {
      ++count;
    }
  }
  return count;
}

bool IsInstructionListedForRoute(StubDecoderRoute route,
                                 std::string_view instruction_name) {
  for (std::string_view listed_instruction :
       GetStubDecoderRouteInstructions(route)) {
    if (listed_instruction == instruction_name) {
      return true;
    }
  }
  return false;
}

bool RouteInstructionListMatchesRouteInfoSequence(StubDecoderRoute route) {
  const auto routed_instructions = GetStubDecoderRouteInstructions(route);
  std::size_t routed_index = 0;
  for (const StubDecoderRouteInfo& route_info : GetStubDecoderRouteInfos()) {
    if (route_info.route != route) {
      continue;
    }
    if (routed_index >= routed_instructions.size() ||
        routed_instructions[routed_index] != route_info.instruction_name) {
      return false;
    }
    ++routed_index;
  }
  return routed_index == routed_instructions.size();
}

bool GlobalRouteInfoSequenceMatchesRouteInstructionLists() {
  const auto route_infos = GetStubDecoderRouteInfos();
  std::size_t route_info_index = 0;
  for (const StubDecoderRouteManifest& manifest : GetStubDecoderRouteManifests()) {
    for (std::string_view instruction_name :
         GetStubDecoderRouteInstructions(manifest.route)) {
      if (route_info_index >= route_infos.size()) {
        return false;
      }
      const StubDecoderRouteInfo& route_info = route_infos[route_info_index];
      if (route_info.route != manifest.route ||
          route_info.instruction_name != instruction_name) {
        return false;
      }
      ++route_info_index;
    }
  }
  return route_info_index == route_infos.size();
}

bool RouteInfoLookupMatchesSequenceEntries() {
  for (const StubDecoderRouteInfo& route_info : GetStubDecoderRouteInfos()) {
    if (FindStubDecoderRouteInfo(route_info.instruction_name) != &route_info) {
      return false;
    }
  }
  return true;
}

bool EntrypointManifestLookupMatchesSequenceEntries() {
  const auto manifests = GetStubDecoderEntrypointManifests();
  for (std::size_t i = 0; i < manifests.size(); ++i) {
    if (FindStubDecoderEntrypointManifest(manifests[i].route) != &manifests[i]) {
      return false;
    }
  }
  return true;
}

bool RouteManifestLookupMatchesSequenceEntries() {
  const auto manifests = GetStubDecoderRouteManifests();
  for (std::size_t i = 0; i < manifests.size(); ++i) {
    if (FindStubDecoderRouteManifest(manifests[i].route) != &manifests[i]) {
      return false;
    }
  }
  return true;
}

bool EntrypointAndRouteManifestSequencesMatchExactly() {
  const auto entrypoint_manifests = GetStubDecoderEntrypointManifests();
  const auto route_manifests = GetStubDecoderRouteManifests();
  if (entrypoint_manifests.size() != route_manifests.size()) {
    return false;
  }
  for (std::size_t i = 0; i < entrypoint_manifests.size(); ++i) {
    const StubDecoderEntrypointManifest& entrypoint_manifest =
        entrypoint_manifests[i];
    const StubDecoderRouteManifest& route_manifest = route_manifests[i];
    if (entrypoint_manifest.route != route_manifest.route ||
        entrypoint_manifest.route_name != route_manifest.route_name ||
        entrypoint_manifest.route_priority != route_manifest.route_priority ||
        entrypoint_manifest.instruction_count !=
            route_manifest.instruction_count) {
      return false;
    }
  }
  return true;
}

bool EntrypointManifestCountsMatchRoutedSurfaces() {
  for (const StubDecoderEntrypointManifest& manifest :
       GetStubDecoderEntrypointManifests()) {
    if (manifest.instruction_count !=
            GetStubDecoderRouteInstructions(manifest.route).size() ||
        manifest.instruction_count != CountRouteInfosForRoute(manifest.route)) {
      return false;
    }
  }
  return true;
}

bool RoutedInstructionNamesFormUniqueBijection() {
  const auto route_infos = GetStubDecoderRouteInfos();
  for (std::size_t i = 0; i < route_infos.size(); ++i) {
    if (!IsInstructionListedForRoute(route_infos[i].route,
                                     route_infos[i].instruction_name)) {
      return false;
    }
    for (std::size_t j = i + 1; j < route_infos.size(); ++j) {
      if (route_infos[i].instruction_name == route_infos[j].instruction_name) {
        return false;
      }
    }
  }

  std::size_t listed_instruction_count = 0;
  for (const StubDecoderRouteManifest& manifest : GetStubDecoderRouteManifests()) {
    for (std::string_view instruction_name :
         GetStubDecoderRouteInstructions(manifest.route)) {
      const StubDecoderRouteInfo* route_info =
          FindStubDecoderRouteInfo(instruction_name);
      if (route_info == nullptr || route_info->route != manifest.route) {
        return false;
      }
      ++listed_instruction_count;
    }
  }

  return listed_instruction_count == route_infos.size();
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

std::uint32_t CountDescriptorsForRoleAndAccess(
    const StubDecodedInstruction& instruction,
    StubOperandRole role,
    StubOperandAccess access) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    const auto& descriptor = instruction.operand_descriptors.descriptors[i];
    if (descriptor.role == role && descriptor.access == access) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountDescriptorsForRoleAndValueClass(
    const StubDecodedInstruction& instruction,
    StubOperandRole role,
    StubOperandValueClass value_class) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    const auto& descriptor = instruction.operand_descriptors.descriptors[i];
    if (descriptor.role == role && descriptor.value_class == value_class) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountDescriptorsForRoleAndFragmentKind(
    const StubDecodedInstruction& instruction,
    StubOperandRole role,
    StubFragmentKind fragment_kind) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    const auto& descriptor = instruction.operand_descriptors.descriptors[i];
    if (descriptor.role == role &&
        descriptor.fragment_shape.kind == fragment_kind) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountDescriptorsForRoleAndComponentCount(
    const StubDecodedInstruction& instruction,
    StubOperandRole role,
    std::uint8_t component_count) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    const auto& descriptor = instruction.operand_descriptors.descriptors[i];
    if (descriptor.role == role &&
        descriptor.component_count == component_count) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountDescriptorsForRoleAndWaveSize(
    const StubDecodedInstruction& instruction,
    StubOperandRole role,
    std::uint8_t wave_size) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    const auto& descriptor = instruction.operand_descriptors.descriptors[i];
    if (descriptor.role == role &&
        descriptor.fragment_shape.wave_size == wave_size) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountDescriptorsForRoleAndElementBitWidth(
    const StubDecodedInstruction& instruction,
    StubOperandRole role,
    std::uint8_t element_bit_width) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    const auto& descriptor = instruction.operand_descriptors.descriptors[i];
    if (descriptor.role == role &&
        descriptor.fragment_shape.element_bit_width == element_bit_width) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountDescriptorsForRoleAndPackedElements(
    const StubDecodedInstruction& instruction,
    StubOperandRole role,
    std::uint8_t packed_elements) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    const auto& descriptor = instruction.operand_descriptors.descriptors[i];
    if (descriptor.role == role &&
        descriptor.fragment_shape.packed_elements == packed_elements) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountDescriptorsForRoleAndDimensions(
    const StubDecodedInstruction& instruction,
    StubOperandRole role,
    std::uint16_t rows,
    std::uint16_t columns,
    std::uint16_t depth) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    const auto& descriptor = instruction.operand_descriptors.descriptors[i];
    if (descriptor.role == role && descriptor.fragment_shape.rows == rows &&
        descriptor.fragment_shape.columns == columns &&
        descriptor.fragment_shape.depth == depth) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountDescriptorsForRoleAndSlotKind(
    const StubDecodedInstruction& instruction,
    StubOperandRole role,
    StubOperandSlotKind slot_kind) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    const auto& descriptor = instruction.operand_descriptors.descriptors[i];
    if (descriptor.role == role && descriptor.slot_kind == slot_kind) {
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

std::uint32_t CountSlotsOfKindWithOutputFlag(
    const StubDecodedInstruction& instruction,
    StubOperandSlotKind slot_kind,
    bool is_output) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    const auto& binding = instruction.operand_slots.bindings[i];
    if (binding.slot_kind == slot_kind && binding.is_output == is_output) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountSlotsOfKindAndValueClass(
    const StubDecodedInstruction& instruction,
    StubOperandSlotKind slot_kind,
    StubOperandValueClass value_class) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    const auto& binding = instruction.operand_slots.bindings[i];
    if (binding.slot_kind == slot_kind && binding.value_class == value_class) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountSlotsOfKindAndFragmentKind(
    const StubDecodedInstruction& instruction,
    StubOperandSlotKind slot_kind,
    StubFragmentKind fragment_kind) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    const auto& binding = instruction.operand_slots.bindings[i];
    if (binding.slot_kind == slot_kind &&
        binding.fragment_shape.kind == fragment_kind) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountSlotsOfKindAndComponentCount(
    const StubDecodedInstruction& instruction,
    StubOperandSlotKind slot_kind,
    std::uint8_t component_count) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    const auto& binding = instruction.operand_slots.bindings[i];
    if (binding.slot_kind == slot_kind &&
        binding.component_count == component_count) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountSlotsOfKindAndWaveSize(
    const StubDecodedInstruction& instruction,
    StubOperandSlotKind slot_kind,
    std::uint8_t wave_size) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    const auto& binding = instruction.operand_slots.bindings[i];
    if (binding.slot_kind == slot_kind &&
        binding.fragment_shape.wave_size == wave_size) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountSlotsOfKindAndElementBitWidth(
    const StubDecodedInstruction& instruction,
    StubOperandSlotKind slot_kind,
    std::uint8_t element_bit_width) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    const auto& binding = instruction.operand_slots.bindings[i];
    if (binding.slot_kind == slot_kind &&
        binding.fragment_shape.element_bit_width == element_bit_width) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountSlotsOfKindAndPackedElements(
    const StubDecodedInstruction& instruction,
    StubOperandSlotKind slot_kind,
    std::uint8_t packed_elements) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    const auto& binding = instruction.operand_slots.bindings[i];
    if (binding.slot_kind == slot_kind &&
        binding.fragment_shape.packed_elements == packed_elements) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountSlotsOfKindAndDimensions(
    const StubDecodedInstruction& instruction,
    StubOperandSlotKind slot_kind,
    std::uint16_t rows,
    std::uint16_t columns,
    std::uint16_t depth) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    const auto& binding = instruction.operand_slots.bindings[i];
    if (binding.slot_kind == slot_kind && binding.fragment_shape.rows == rows &&
        binding.fragment_shape.columns == columns &&
        binding.fragment_shape.depth == depth) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountSlotsOfKindAndLogicalOperandIndex(
    const StubDecodedInstruction& instruction,
    StubOperandSlotKind slot_kind,
    std::uint32_t logical_operand_index) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    const auto& binding = instruction.operand_slots.bindings[i];
    if (binding.slot_kind == slot_kind &&
        binding.logical_operand_index == logical_operand_index) {
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

std::uint32_t CountSlotsWithValueClass(const StubDecodedInstruction& instruction,
                                       StubOperandValueClass value_class) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    if (instruction.operand_slots.bindings[i].value_class == value_class) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountDescriptorsWithValueClass(
    const StubDecodedInstruction& instruction,
    StubOperandValueClass value_class) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    if (instruction.operand_descriptors.descriptors[i].value_class ==
        value_class) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountSlotsWithValueClassAndComponentCount(
    const StubDecodedInstruction& instruction,
    StubOperandValueClass value_class,
    std::uint8_t component_count) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    const auto& binding = instruction.operand_slots.bindings[i];
    if (binding.value_class == value_class &&
        binding.component_count == component_count) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountDescriptorsWithValueClassAndComponentCount(
    const StubDecodedInstruction& instruction,
    StubOperandValueClass value_class,
    std::uint8_t component_count) {
  std::uint32_t count = 0;
  for (std::uint32_t i = 0; i < instruction.operand_descriptors.descriptor_count;
       ++i) {
    const auto& descriptor = instruction.operand_descriptors.descriptors[i];
    if (descriptor.value_class == value_class &&
        descriptor.component_count == component_count) {
      ++count;
    }
  }
  return count;
}

ShapeExtents FindUniqueSlotShapeExtents(const StubDecodedInstruction& instruction,
                                        StubOperandSlotKind slot_kind) {
  ShapeExtents extents;
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    const auto& binding = instruction.operand_slots.bindings[i];
    if (binding.slot_kind != slot_kind) {
      continue;
    }
    if (!extents.valid) {
      extents.rows = binding.fragment_shape.rows;
      extents.columns = binding.fragment_shape.columns;
      extents.depth = binding.fragment_shape.depth;
      extents.valid = true;
      continue;
    }
    if (binding.fragment_shape.rows != extents.rows ||
        binding.fragment_shape.columns != extents.columns ||
        binding.fragment_shape.depth != extents.depth) {
      return {};
    }
  }
  return extents;
}

std::uint8_t FindUniqueSlotElementBitWidth(
    const StubDecodedInstruction& instruction,
    StubOperandSlotKind slot_kind) {
  bool found = false;
  std::uint8_t element_bit_width = 0;
  for (std::uint32_t i = 0; i < instruction.operand_slots.binding_count; ++i) {
    const auto& binding = instruction.operand_slots.bindings[i];
    if (binding.slot_kind != slot_kind) {
      continue;
    }
    if (!found) {
      found = true;
      element_bit_width = binding.fragment_shape.element_bit_width;
      continue;
    }
    if (binding.fragment_shape.element_bit_width != element_bit_width) {
      return 0xff;
    }
  }
  return found ? element_bit_width : 0xff;
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

  for (std::string_view instruction_name :
       GetStubDecoderRouteInstructions(StubDecoderRoute::kVop3p)) {
    if (instruction_name.rfind("V_PK_", 0) != 0 ||
        instruction_name.find("BF16") == std::string_view::npos) {
      continue;
    }
    const StubDecodedInstruction decoded = DecodeVop3pStub(instruction_name);
    const StubDecoderRouteInfo* route_info =
        FindStubDecoderRouteInfo(instruction_name);
    if (!Expect(decoded.status == StubDecodeStatus::kDecodedStub,
                "expected routed packed VOP3P seed to decode")) {
      return 1;
    }
    if (!Expect(route_info != nullptr &&
                    MatchesRouteInfoPayload(decoded, *route_info),
                "expected routed packed VOP3P seed to preserve route-info and RDNA4 provenance")) {
      return 1;
    }
    if (!Expect(
            MatchesRouteMetadata(
                decoded,
                {StubDecoderRoute::kVop3p, "kVop3p", "DecodeVop3pStub", 1}),
            "expected routed packed VOP3P seed to keep exact route metadata")) {
      return 1;
    }
    if (!Expect(
            decoded.execution_domain == StubExecutionDomain::kVectorAlu &&
                decoded.opcode_shape ==
                    (instruction_name == "V_PK_FMA_BF16"
                         ? StubOpcodeShape::kVop3pPackedFma
                         : StubOpcodeShape::kVop3pPackedBinary) &&
                !decoded.uses_accumulator && !decoded.uses_tensor_memory &&
                !decoded.uses_scale_path && decoded.uses_paired_operands &&
                MatchesLayout(
                    decoded,
                    {instruction_name == "V_PK_ADD_BF16"
                         ? StubOperandLayoutKind::kPkAddBf16
                         : instruction_name == "V_PK_FMA_BF16"
                               ? StubOperandLayoutKind::kPkFmaBf16
                               : instruction_name == "V_PK_MUL_BF16"
                                     ? StubOperandLayoutKind::kPkMulBf16
                                     : instruction_name == "V_PK_MIN_NUM_BF16"
                                           ? StubOperandLayoutKind::kPkMinNumBf16
                                           : StubOperandLayoutKind::kPkMaxNumBf16,
                     instruction_name == "V_PK_FMA_BF16" ? 3u : 2u,
                     1,
                     0,
                     false,
                     false,
                     false,
                     false,
                     false}),
            "expected routed packed VOP3P seed to keep exact top-level route/layout metadata")) {
      return 1;
    }
    if (!Expect(MatchesTopLevelFlags(decoded, false, false, false, true),
                "expected routed packed VOP3P seed to keep exact top-level flag composition")) {
      return 1;
    }
    if (!Expect(MatchesLayoutToRecordInvariants(decoded),
                "expected routed packed VOP3P seed to keep exact layout-to-record consistency")) {
      return 1;
    }
    if (!Expect(MatchesDescriptorToSlotParity(decoded),
                "expected routed packed VOP3P seed to keep exact descriptor-to-slot parity")) {
      return 1;
    }
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
    const StubDecoderRouteInfo* route_info =
        FindStubDecoderRouteInfo(instruction_name);
    if (!Expect(decoded.status == StubDecodeStatus::kDecodedStub,
                "expected routed WMMA/SWMMAC seed to decode")) {
      return 1;
    }
    if (!Expect(route_info != nullptr &&
                    MatchesRouteInfoPayload(decoded, *route_info),
                "expected routed WMMA/SWMMAC seed to preserve route-info and RDNA4 provenance")) {
      return 1;
    }
    if (!Expect(
            MatchesRouteMetadata(
                decoded,
                {StubDecoderRoute::kVop3p, "kVop3p", "DecodeVop3pStub", 1}),
            "expected routed WMMA/SWMMAC seed to keep exact route metadata")) {
      return 1;
    }
    if (!Expect(HasMatrixSlot(decoded) && HasMatrixDescriptor(decoded),
                "expected routed WMMA/SWMMAC seed to materialize matrix metadata")) {
      return 1;
    }
    if (!Expect(MatchesLayoutToRecordInvariants(decoded),
                "expected routed WMMA/SWMMAC seed to keep exact layout-to-record consistency")) {
      return 1;
    }
    if (!Expect(MatchesDescriptorToSlotParity(decoded),
                "expected routed WMMA/SWMMAC seed to keep exact descriptor-to-slot parity")) {
      return 1;
    }
    if (decoded.uses_scale_path) {
      if (!Expect(
              decoded.execution_domain == StubExecutionDomain::kMatrix &&
                  decoded.opcode_shape == StubOpcodeShape::kWmmaScale &&
                  decoded.operand_layout.source_count == 3 &&
                  decoded.operand_layout.destination_count == 1 &&
                  decoded.operand_layout.accumulator_source_count == 1 &&
                  decoded.operand_layout.has_scale_operand &&
                  !decoded.operand_layout.has_paired_scale_operand &&
                  !decoded.operand_layout.has_tensor_descriptor &&
                  !decoded.operand_layout.touches_lds &&
                  !decoded.operand_layout.is_store,
              "expected routed WMMA scale seed to keep exact top-level route/layout metadata")) {
        return 1;
      }
      if (!Expect(MatchesTopLevelFlags(decoded, true, false, true, false),
                  "expected routed WMMA scale seed to keep exact top-level flag composition")) {
        return 1;
      }
    } else {
      if (!Expect(
              decoded.execution_domain == StubExecutionDomain::kMatrix &&
                  decoded.opcode_shape ==
                      (instruction_name.rfind("V_SWMMAC_", 0) == 0
                           ? StubOpcodeShape::kSwmmacCore
                           : StubOpcodeShape::kWmmaCore) &&
                  decoded.operand_layout.source_count == 2 &&
                  decoded.operand_layout.destination_count == 1 &&
                  decoded.operand_layout.accumulator_source_count == 1 &&
                  !decoded.operand_layout.has_scale_operand &&
                  !decoded.operand_layout.has_paired_scale_operand &&
                  !decoded.operand_layout.has_tensor_descriptor &&
                  !decoded.operand_layout.touches_lds &&
                  !decoded.operand_layout.is_store,
              "expected routed WMMA/SWMMAC core seed to keep exact top-level route/layout metadata")) {
        return 1;
      }
      if (!Expect(MatchesTopLevelFlags(decoded, true, false, false, false),
                  "expected routed WMMA/SWMMAC core seed to keep exact top-level flag composition")) {
        return 1;
      }
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
                    CountSlotsWithValueClass(
                        decoded, StubOperandValueClass::kMatrixFragment) == 3 &&
                    CountSlotsWithValueClass(
                        decoded, StubOperandValueClass::kAccumulatorFragment) == 1 &&
                    CountSlotsWithValueClassAndComponentCount(
                        decoded, StubOperandValueClass::kMatrixFragment, 1) == 3 &&
                    CountSlotsWithValueClassAndComponentCount(
                        decoded, StubOperandValueClass::kAccumulatorFragment, 1) == 1 &&
                    CountDescriptorsWithValueClass(
                        decoded, StubOperandValueClass::kMatrixFragment) == 3 &&
                    CountDescriptorsWithValueClass(
                        decoded, StubOperandValueClass::kAccumulatorFragment) == 1 &&
                    CountDescriptorsWithValueClassAndComponentCount(
                        decoded, StubOperandValueClass::kMatrixFragment, 1) == 3 &&
                    CountDescriptorsWithValueClassAndComponentCount(
                        decoded, StubOperandValueClass::kAccumulatorFragment,
                        1) == 1 &&
                    AllMatrixSlotsHaveWaveSize(decoded, 32) &&
                    AllMatrixDescriptorsHaveWaveSize(decoded, 32),
                "expected routed WMMA/SWMMAC matrix fragments to stay wave32")) {
      return 1;
    }
    if (instruction_name.rfind("V_WMMA_SCALE", 0) == 0 &&
        instruction_name.rfind("V_WMMA_LD_SCALE", 0) != 0) {
      if (!Expect(decoded.uses_scale_path &&
                      !decoded.uses_tensor_memory &&
                      !decoded.uses_paired_operands &&
                      decoded.operand_slots.binding_count == 5 &&
                      decoded.operand_descriptors.descriptor_count == 5 &&
                      CountOutputSlots(decoded) == 1 &&
                      CountDescriptorsWithAccess(decoded,
                                                 StubOperandAccess::kRead) == 4 &&
                      CountDescriptorsWithAccess(decoded,
                                                 StubOperandAccess::kWrite) == 1 &&
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
                      CountDescriptorsForRoleAndAccess(
                          decoded, StubOperandRole::kDestination,
                          StubOperandAccess::kWrite) == 1 &&
                      CountDescriptorsForRoleAndAccess(
                          decoded, StubOperandRole::kSource0,
                          StubOperandAccess::kRead) == 1 &&
                      CountDescriptorsForRoleAndAccess(
                          decoded, StubOperandRole::kSource1,
                          StubOperandAccess::kRead) == 1 &&
                      CountDescriptorsForRoleAndAccess(
                          decoded, StubOperandRole::kAccumulator,
                          StubOperandAccess::kRead) == 1 &&
                    CountDescriptorsForRoleAndAccess(
                        decoded, StubOperandRole::kScale,
                        StubOperandAccess::kRead) == 1 &&
                      CountDescriptorsForRoleAndValueClass(
                          decoded, StubOperandRole::kDestination,
                          StubOperandValueClass::kMatrixFragment) == 1 &&
                      CountDescriptorsForRoleAndValueClass(
                          decoded, StubOperandRole::kSource0,
                          StubOperandValueClass::kMatrixFragment) == 1 &&
                      CountDescriptorsForRoleAndValueClass(
                          decoded, StubOperandRole::kSource1,
                          StubOperandValueClass::kMatrixFragment) == 1 &&
                      CountDescriptorsForRoleAndValueClass(
                          decoded, StubOperandRole::kAccumulator,
                          StubOperandValueClass::kAccumulatorFragment) == 1 &&
                      CountDescriptorsForRoleAndValueClass(
                          decoded, StubOperandRole::kScale,
                          StubOperandValueClass::kScalarRegister) == 1 &&
                      CountDescriptorsForRoleAndFragmentKind(
                          decoded, StubOperandRole::kDestination,
                          StubFragmentKind::kMatrix) == 1 &&
                      CountDescriptorsForRoleAndFragmentKind(
                          decoded, StubOperandRole::kSource0,
                          StubFragmentKind::kMatrix) == 1 &&
                      CountDescriptorsForRoleAndFragmentKind(
                          decoded, StubOperandRole::kSource1,
                          StubFragmentKind::kMatrix) == 1 &&
                      CountDescriptorsForRoleAndFragmentKind(
                          decoded, StubOperandRole::kAccumulator,
                          StubFragmentKind::kMatrix) == 1 &&
                      CountDescriptorsForRoleAndFragmentKind(
                          decoded, StubOperandRole::kScale,
                          StubFragmentKind::kScalar) == 1 &&
                      CountDescriptorsForRoleAndComponentCount(
                          decoded, StubOperandRole::kDestination, 1) == 1 &&
                      CountDescriptorsForRoleAndComponentCount(
                          decoded, StubOperandRole::kSource0, 1) == 1 &&
                      CountDescriptorsForRoleAndComponentCount(
                          decoded, StubOperandRole::kSource1, 1) == 1 &&
                      CountDescriptorsForRoleAndComponentCount(
                          decoded, StubOperandRole::kAccumulator, 1) == 1 &&
                      CountDescriptorsForRoleAndComponentCount(
                          decoded, StubOperandRole::kScale, 1) == 1 &&
                      CountSlotsOfKindAndValueClass(
                          decoded, StubOperandSlotKind::kDestination,
                          StubOperandValueClass::kMatrixFragment) == 1 &&
                      CountSlotsOfKindAndValueClass(
                          decoded, StubOperandSlotKind::kSource0,
                          StubOperandValueClass::kMatrixFragment) == 1 &&
                      CountSlotsOfKindAndValueClass(
                          decoded, StubOperandSlotKind::kSource1,
                          StubOperandValueClass::kMatrixFragment) == 1 &&
                      CountSlotsOfKindAndValueClass(
                          decoded, StubOperandSlotKind::kAccumulatorSource,
                          StubOperandValueClass::kAccumulatorFragment) == 1 &&
                      CountSlotsOfKindAndValueClass(
                          decoded, StubOperandSlotKind::kScaleSource,
                          StubOperandValueClass::kScalarRegister) == 1 &&
                      CountSlotsOfKindAndFragmentKind(
                          decoded, StubOperandSlotKind::kDestination,
                          StubFragmentKind::kMatrix) == 1 &&
                      CountSlotsOfKindAndFragmentKind(
                          decoded, StubOperandSlotKind::kSource0,
                          StubFragmentKind::kMatrix) == 1 &&
                      CountSlotsOfKindAndFragmentKind(
                          decoded, StubOperandSlotKind::kSource1,
                          StubFragmentKind::kMatrix) == 1 &&
                      CountSlotsOfKindAndFragmentKind(
                          decoded, StubOperandSlotKind::kAccumulatorSource,
                          StubFragmentKind::kMatrix) == 1 &&
                      CountSlotsOfKindAndFragmentKind(
                          decoded, StubOperandSlotKind::kScaleSource,
                          StubFragmentKind::kScalar) == 1 &&
                      CountSlotsOfKindAndComponentCount(
                          decoded, StubOperandSlotKind::kDestination, 1) == 1 &&
                      CountSlotsOfKindAndComponentCount(
                          decoded, StubOperandSlotKind::kSource0, 1) == 1 &&
                      CountSlotsOfKindAndComponentCount(
                          decoded, StubOperandSlotKind::kSource1, 1) == 1 &&
                      CountSlotsOfKindAndComponentCount(
                          decoded, StubOperandSlotKind::kAccumulatorSource, 1) == 1 &&
                      CountSlotsOfKindAndComponentCount(
                          decoded, StubOperandSlotKind::kScaleSource, 1) == 1 &&
                      CountSlotsOfKindWithOutputFlag(
                          decoded, StubOperandSlotKind::kDestination, true) == 1 &&
                      CountSlotsOfKindWithOutputFlag(
                          decoded, StubOperandSlotKind::kSource0, false) == 1 &&
                      CountSlotsOfKindWithOutputFlag(
                          decoded, StubOperandSlotKind::kSource1, false) == 1 &&
                      CountSlotsOfKindWithOutputFlag(
                          decoded, StubOperandSlotKind::kAccumulatorSource,
                          false) == 1 &&
                      CountSlotsOfKindWithOutputFlag(
                          decoded, StubOperandSlotKind::kScaleSource, false) == 1 &&
                      CountSlotsWithFragmentKindAndWaveSize(
                          decoded, StubFragmentKind::kMatrix, 32) == 4 &&
                      CountDescriptorsWithFragmentKindAndWaveSize(
                          decoded, StubFragmentKind::kMatrix, 32) == 4 &&
                      CountSlotsWithFragmentKindAndWaveSize(
                          decoded, StubFragmentKind::kScalar, 0) == 1 &&
                      CountDescriptorsWithFragmentKindAndWaveSize(
                          decoded, StubFragmentKind::kScalar, 0) == 1 &&
                      CountSlotsWithValueClass(
                          decoded, StubOperandValueClass::kMatrixFragment) == 3 &&
                      CountSlotsWithValueClass(
                          decoded,
                          StubOperandValueClass::kAccumulatorFragment) == 1 &&
                      CountSlotsWithValueClass(
                          decoded, StubOperandValueClass::kScalarRegister) == 1 &&
                      CountSlotsWithValueClassAndComponentCount(
                          decoded, StubOperandValueClass::kMatrixFragment, 1) == 3 &&
                      CountSlotsWithValueClassAndComponentCount(
                          decoded,
                          StubOperandValueClass::kAccumulatorFragment, 1) == 1 &&
                      CountSlotsWithValueClassAndComponentCount(
                          decoded, StubOperandValueClass::kScalarRegister, 1) == 1 &&
                      CountDescriptorsWithValueClass(
                          decoded, StubOperandValueClass::kMatrixFragment) == 3 &&
                      CountDescriptorsWithValueClass(
                          decoded,
                          StubOperandValueClass::kAccumulatorFragment) == 1 &&
                      CountDescriptorsWithValueClass(
                          decoded, StubOperandValueClass::kScalarRegister) == 1 &&
                      CountDescriptorsWithValueClassAndComponentCount(
                          decoded, StubOperandValueClass::kMatrixFragment, 1) == 3 &&
                      CountDescriptorsWithValueClassAndComponentCount(
                          decoded,
                          StubOperandValueClass::kAccumulatorFragment, 1) == 1 &&
                      CountDescriptorsWithValueClassAndComponentCount(
                          decoded, StubOperandValueClass::kScalarRegister, 1) == 1 &&
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
      if (!Expect(
              CountDescriptorsForRoleAndSlotKind(
                  decoded, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination) == 1 &&
                  CountDescriptorsForRoleAndSlotKind(
                      decoded, StubOperandRole::kSource0,
                      StubOperandSlotKind::kSource0) == 1 &&
                  CountDescriptorsForRoleAndSlotKind(
                      decoded, StubOperandRole::kSource1,
                      StubOperandSlotKind::kSource1) == 1 &&
                  CountDescriptorsForRoleAndSlotKind(
                      decoded, StubOperandRole::kAccumulator,
                      StubOperandSlotKind::kAccumulatorSource) == 1 &&
                  CountDescriptorsForRoleAndSlotKind(
                      decoded, StubOperandRole::kScale,
                      StubOperandSlotKind::kScaleSource) == 1 &&
                  CountSlotsOfKindAndLogicalOperandIndex(
                      decoded, StubOperandSlotKind::kDestination, 0) == 1 &&
                  CountSlotsOfKindAndLogicalOperandIndex(
                      decoded, StubOperandSlotKind::kSource0, 1) == 1 &&
                  CountSlotsOfKindAndLogicalOperandIndex(
                      decoded, StubOperandSlotKind::kSource1, 2) == 1 &&
                  CountSlotsOfKindAndLogicalOperandIndex(
                      decoded, StubOperandSlotKind::kAccumulatorSource, 3) ==
                      1 &&
                  CountSlotsOfKindAndLogicalOperandIndex(
                      decoded, StubOperandSlotKind::kScaleSource, 4) == 1,
              "expected routed WMMA scale seed to keep exact role/slot and logical-index mapping")) {
        return 1;
      }
      if (!Expect(
              CountDescriptorsForRoleAndWaveSize(
                  decoded, StubOperandRole::kDestination, 32) == 1 &&
                  CountDescriptorsForRoleAndWaveSize(
                      decoded, StubOperandRole::kSource0, 32) == 1 &&
                  CountDescriptorsForRoleAndWaveSize(
                      decoded, StubOperandRole::kSource1, 32) == 1 &&
                  CountDescriptorsForRoleAndWaveSize(
                      decoded, StubOperandRole::kAccumulator, 32) == 1 &&
                  CountDescriptorsForRoleAndWaveSize(
                      decoded, StubOperandRole::kScale, 0) == 1 &&
                  CountSlotsOfKindAndWaveSize(
                      decoded, StubOperandSlotKind::kDestination, 32) == 1 &&
                  CountSlotsOfKindAndWaveSize(
                      decoded, StubOperandSlotKind::kSource0, 32) == 1 &&
                  CountSlotsOfKindAndWaveSize(
                      decoded, StubOperandSlotKind::kSource1, 32) == 1 &&
                  CountSlotsOfKindAndWaveSize(
                      decoded, StubOperandSlotKind::kAccumulatorSource, 32) ==
                      1 &&
                  CountSlotsOfKindAndWaveSize(
                      decoded, StubOperandSlotKind::kScaleSource, 0) == 1,
              "expected routed WMMA scale seed to keep exact role/slot wave-size mapping")) {
        return 1;
      }
      const std::uint8_t scale_destination_element_bit_width =
          FindUniqueSlotElementBitWidth(decoded, StubOperandSlotKind::kDestination);
      const std::uint8_t scale_source0_element_bit_width =
          FindUniqueSlotElementBitWidth(decoded, StubOperandSlotKind::kSource0);
      const std::uint8_t scale_source1_element_bit_width =
          FindUniqueSlotElementBitWidth(decoded, StubOperandSlotKind::kSource1);
      const std::uint8_t scale_accumulator_element_bit_width =
          FindUniqueSlotElementBitWidth(decoded, StubOperandSlotKind::kAccumulatorSource);
      const std::uint8_t scale_scale_element_bit_width =
          FindUniqueSlotElementBitWidth(decoded, StubOperandSlotKind::kScaleSource);
      if (!Expect(
              scale_destination_element_bit_width != 0xff &&
                  scale_source0_element_bit_width != 0xff &&
                  scale_source1_element_bit_width != 0xff &&
                  scale_accumulator_element_bit_width != 0xff &&
                  scale_scale_element_bit_width == 32 &&
                  scale_source0_element_bit_width ==
                      scale_source1_element_bit_width &&
                  scale_destination_element_bit_width ==
                      scale_accumulator_element_bit_width &&
                  CountDescriptorsForRoleAndElementBitWidth(
                      decoded, StubOperandRole::kDestination,
                      scale_destination_element_bit_width) == 1 &&
                  CountDescriptorsForRoleAndElementBitWidth(
                      decoded, StubOperandRole::kSource0,
                      scale_source0_element_bit_width) == 1 &&
                  CountDescriptorsForRoleAndElementBitWidth(
                      decoded, StubOperandRole::kSource1,
                      scale_source1_element_bit_width) == 1 &&
                  CountDescriptorsForRoleAndElementBitWidth(
                      decoded, StubOperandRole::kAccumulator,
                      scale_accumulator_element_bit_width) == 1 &&
                  CountDescriptorsForRoleAndElementBitWidth(
                      decoded, StubOperandRole::kScale,
                      scale_scale_element_bit_width) == 1 &&
                  CountSlotsOfKindAndElementBitWidth(
                      decoded, StubOperandSlotKind::kDestination,
                      scale_destination_element_bit_width) == 1 &&
                  CountSlotsOfKindAndElementBitWidth(
                      decoded, StubOperandSlotKind::kSource0,
                      scale_source0_element_bit_width) == 1 &&
                  CountSlotsOfKindAndElementBitWidth(
                      decoded, StubOperandSlotKind::kSource1,
                      scale_source1_element_bit_width) == 1 &&
                  CountSlotsOfKindAndElementBitWidth(
                      decoded, StubOperandSlotKind::kAccumulatorSource,
                      scale_accumulator_element_bit_width) == 1 &&
                  CountSlotsOfKindAndElementBitWidth(
                      decoded, StubOperandSlotKind::kScaleSource,
                      scale_scale_element_bit_width) == 1,
              "expected routed WMMA scale seed to keep exact role/slot element-width mapping")) {
        return 1;
      }
      if (!Expect(
              CountDescriptorsForRoleAndPackedElements(
                  decoded, StubOperandRole::kDestination, 0) == 1 &&
                  CountDescriptorsForRoleAndPackedElements(
                      decoded, StubOperandRole::kSource0, 0) == 1 &&
                  CountDescriptorsForRoleAndPackedElements(
                      decoded, StubOperandRole::kSource1, 0) == 1 &&
                  CountDescriptorsForRoleAndPackedElements(
                      decoded, StubOperandRole::kAccumulator, 0) == 1 &&
                  CountDescriptorsForRoleAndPackedElements(
                      decoded, StubOperandRole::kScale, 1) == 1 &&
                  CountSlotsOfKindAndPackedElements(
                      decoded, StubOperandSlotKind::kDestination, 0) == 1 &&
                  CountSlotsOfKindAndPackedElements(
                      decoded, StubOperandSlotKind::kSource0, 0) == 1 &&
                  CountSlotsOfKindAndPackedElements(
                      decoded, StubOperandSlotKind::kSource1, 0) == 1 &&
                  CountSlotsOfKindAndPackedElements(
                      decoded, StubOperandSlotKind::kAccumulatorSource, 0) ==
                      1 &&
                  CountSlotsOfKindAndPackedElements(
                      decoded, StubOperandSlotKind::kScaleSource, 1) == 1,
              "expected routed WMMA scale seed to keep exact role/slot packed-elements mapping")) {
        return 1;
      }
      const ShapeExtents scale_destination_extents =
          FindUniqueSlotShapeExtents(decoded, StubOperandSlotKind::kDestination);
      const ShapeExtents scale_source0_extents =
          FindUniqueSlotShapeExtents(decoded, StubOperandSlotKind::kSource0);
      const ShapeExtents scale_source1_extents =
          FindUniqueSlotShapeExtents(decoded, StubOperandSlotKind::kSource1);
      const ShapeExtents scale_accumulator_extents =
          FindUniqueSlotShapeExtents(decoded, StubOperandSlotKind::kAccumulatorSource);
      const ShapeExtents scale_scale_extents =
          FindUniqueSlotShapeExtents(decoded, StubOperandSlotKind::kScaleSource);
      if (!Expect(
              scale_destination_extents.valid && scale_source0_extents.valid &&
                  scale_source1_extents.valid && scale_accumulator_extents.valid &&
                  scale_scale_extents.valid &&
                  scale_destination_extents.rows == scale_source0_extents.rows &&
                  scale_destination_extents.columns ==
                      scale_source0_extents.columns &&
                  scale_destination_extents.depth ==
                      scale_source0_extents.depth &&
                  scale_destination_extents.rows == scale_source1_extents.rows &&
                  scale_destination_extents.columns ==
                      scale_source1_extents.columns &&
                  scale_destination_extents.depth ==
                      scale_source1_extents.depth &&
                  scale_destination_extents.rows ==
                      scale_accumulator_extents.rows &&
                  scale_destination_extents.columns ==
                      scale_accumulator_extents.columns &&
                  scale_destination_extents.depth ==
                      scale_accumulator_extents.depth &&
                  scale_scale_extents.rows == 1 &&
                  scale_scale_extents.columns == 1 &&
                  scale_scale_extents.depth == 1 &&
                  CountDescriptorsForRoleAndDimensions(
                      decoded, StubOperandRole::kDestination,
                      scale_destination_extents.rows,
                      scale_destination_extents.columns,
                      scale_destination_extents.depth) == 1 &&
                  CountDescriptorsForRoleAndDimensions(
                      decoded, StubOperandRole::kSource0,
                      scale_source0_extents.rows, scale_source0_extents.columns,
                      scale_source0_extents.depth) == 1 &&
                  CountDescriptorsForRoleAndDimensions(
                      decoded, StubOperandRole::kSource1,
                      scale_source1_extents.rows, scale_source1_extents.columns,
                      scale_source1_extents.depth) == 1 &&
                  CountDescriptorsForRoleAndDimensions(
                      decoded, StubOperandRole::kAccumulator,
                      scale_accumulator_extents.rows,
                      scale_accumulator_extents.columns,
                      scale_accumulator_extents.depth) == 1 &&
                  CountDescriptorsForRoleAndDimensions(
                      decoded, StubOperandRole::kScale,
                      scale_scale_extents.rows, scale_scale_extents.columns,
                      scale_scale_extents.depth) == 1 &&
                  CountSlotsOfKindAndDimensions(
                      decoded, StubOperandSlotKind::kDestination,
                      scale_destination_extents.rows,
                      scale_destination_extents.columns,
                      scale_destination_extents.depth) == 1 &&
                  CountSlotsOfKindAndDimensions(
                      decoded, StubOperandSlotKind::kSource0,
                      scale_source0_extents.rows, scale_source0_extents.columns,
                      scale_source0_extents.depth) == 1 &&
                  CountSlotsOfKindAndDimensions(
                      decoded, StubOperandSlotKind::kSource1,
                      scale_source1_extents.rows, scale_source1_extents.columns,
                      scale_source1_extents.depth) == 1 &&
                  CountSlotsOfKindAndDimensions(
                      decoded, StubOperandSlotKind::kAccumulatorSource,
                      scale_accumulator_extents.rows,
                      scale_accumulator_extents.columns,
                      scale_accumulator_extents.depth) == 1 &&
                  CountSlotsOfKindAndDimensions(
                      decoded, StubOperandSlotKind::kScaleSource,
                      scale_scale_extents.rows, scale_scale_extents.columns,
                      scale_scale_extents.depth) == 1,
              "expected routed WMMA scale seed to keep exact role/slot dimension mapping")) {
        return 1;
      }
      if (!Expect(
              CountRoleBindings(decoded, StubOperandRole::kDestination) == 1 &&
                  CountRoleBindings(decoded, StubOperandRole::kSource0) == 1 &&
                  CountRoleBindings(decoded, StubOperandRole::kSource1) == 1 &&
                  CountRoleBindings(decoded, StubOperandRole::kAccumulator) == 1 &&
                  CountRoleBindings(decoded, StubOperandRole::kScale) == 1 &&
                  CountRoleBindingsWithCount(decoded, StubOperandRole::kDestination,
                                             1) == 1 &&
                  CountRoleBindingsWithCount(decoded, StubOperandRole::kSource0,
                                             1) == 1 &&
                  CountRoleBindingsWithCount(decoded, StubOperandRole::kSource1,
                                             1) == 1 &&
                  CountRoleBindingsWithCount(decoded,
                                             StubOperandRole::kAccumulator,
                                             1) == 1 &&
                  CountRoleBindingsWithCount(decoded, StubOperandRole::kScale,
                                             1) == 1 &&
                  CountRoleBindingsWithOutputFlag(
                      decoded, StubOperandRole::kDestination, true) == 1 &&
                  CountRoleBindingsWithOutputFlag(
                      decoded, StubOperandRole::kSource0, false) == 1 &&
                  CountRoleBindingsWithOutputFlag(
                      decoded, StubOperandRole::kSource1, false) == 1 &&
                  CountRoleBindingsWithOutputFlag(
                      decoded, StubOperandRole::kAccumulator, false) == 1 &&
                  CountRoleBindingsWithOutputFlag(
                      decoded, StubOperandRole::kScale, false) == 1,
              "expected routed WMMA scale seed to keep exact operand-role binding mapping")) {
        return 1;
      }
      if (!Expect(decoded.operand_roles.binding_count == 5 &&
                      AllRoleBindingsExplicit(decoded),
                  "expected routed WMMA scale seed to keep exact operand-role binding-count and explicitness")) {
        return 1;
      }
      if (!Expect(
              MatchesRoleBindingSequence(
                  decoded,
                  {{StubOperandRole::kSource0, 1, false, false},
                   {StubOperandRole::kSource1, 1, false, false},
                   {StubOperandRole::kAccumulator, 1, false, false},
                   {StubOperandRole::kScale, 1, false, false},
                   {StubOperandRole::kDestination, 1, true, false}}) &&
                  MatchesSlotBindingSequence(
                      decoded,
                      {{StubOperandSlotKind::kDestination,
                        StubOperandValueClass::kMatrixFragment,
                        0,
                        1,
                        true,
                        false},
                       {StubOperandSlotKind::kSource0,
                        StubOperandValueClass::kMatrixFragment,
                        1,
                        1,
                        false,
                        false},
                       {StubOperandSlotKind::kSource1,
                        StubOperandValueClass::kMatrixFragment,
                        2,
                        1,
                        false,
                        false},
                       {StubOperandSlotKind::kAccumulatorSource,
                        StubOperandValueClass::kAccumulatorFragment,
                        3,
                        1,
                        false,
                        false},
                       {StubOperandSlotKind::kScaleSource,
                        StubOperandValueClass::kScalarRegister,
                        4,
                        1,
                        false,
                        false}}) &&
                  MatchesDescriptorSequence(
                      decoded,
                      {{StubOperandRole::kDestination,
                        StubOperandSlotKind::kDestination,
                        StubOperandValueClass::kMatrixFragment,
                        StubOperandAccess::kWrite,
                        1,
                        false},
                       {StubOperandRole::kSource0,
                        StubOperandSlotKind::kSource0,
                        StubOperandValueClass::kMatrixFragment,
                        StubOperandAccess::kRead,
                        1,
                        false},
                       {StubOperandRole::kSource1,
                        StubOperandSlotKind::kSource1,
                        StubOperandValueClass::kMatrixFragment,
                        StubOperandAccess::kRead,
                        1,
                        false},
                       {StubOperandRole::kAccumulator,
                        StubOperandSlotKind::kAccumulatorSource,
                        StubOperandValueClass::kAccumulatorFragment,
                        StubOperandAccess::kRead,
                        1,
                        false},
                       {StubOperandRole::kScale,
                        StubOperandSlotKind::kScaleSource,
                        StubOperandValueClass::kScalarRegister,
                        StubOperandAccess::kRead,
                        1,
                        false}}),
              "expected routed WMMA scale seed to keep exact operand-role, slot, and descriptor order")) {
        return 1;
      }
    } else {
      if (!Expect(decoded.operand_slots.binding_count == 4 &&
                      decoded.operand_descriptors.descriptor_count == 4 &&
                      !decoded.uses_scale_path &&
                      !decoded.uses_tensor_memory &&
                      !decoded.uses_paired_operands &&
                      CountOutputSlots(decoded) == 1 &&
                      CountDescriptorsWithAccess(decoded,
                                                 StubOperandAccess::kRead) == 3 &&
                      CountDescriptorsWithAccess(decoded,
                                                 StubOperandAccess::kWrite) == 1 &&
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
                      CountDescriptorsForRoleAndAccess(
                          decoded, StubOperandRole::kDestination,
                          StubOperandAccess::kWrite) == 1 &&
                      CountDescriptorsForRoleAndAccess(
                          decoded, StubOperandRole::kSource0,
                          StubOperandAccess::kRead) == 1 &&
                      CountDescriptorsForRoleAndAccess(
                          decoded, StubOperandRole::kSource1,
                          StubOperandAccess::kRead) == 1 &&
                      CountDescriptorsForRoleAndAccess(
                          decoded, StubOperandRole::kAccumulator,
                          StubOperandAccess::kRead) == 1 &&
                      CountDescriptorsForRoleAndValueClass(
                          decoded, StubOperandRole::kDestination,
                          StubOperandValueClass::kMatrixFragment) == 1 &&
                      CountDescriptorsForRoleAndValueClass(
                          decoded, StubOperandRole::kSource0,
                          StubOperandValueClass::kMatrixFragment) == 1 &&
                      CountDescriptorsForRoleAndValueClass(
                          decoded, StubOperandRole::kSource1,
                          StubOperandValueClass::kMatrixFragment) == 1 &&
                      CountDescriptorsForRoleAndValueClass(
                          decoded, StubOperandRole::kAccumulator,
                          StubOperandValueClass::kAccumulatorFragment) == 1 &&
                      CountDescriptorsForRoleAndFragmentKind(
                          decoded, StubOperandRole::kDestination,
                          StubFragmentKind::kMatrix) == 1 &&
                      CountDescriptorsForRoleAndFragmentKind(
                          decoded, StubOperandRole::kSource0,
                          StubFragmentKind::kMatrix) == 1 &&
                      CountDescriptorsForRoleAndFragmentKind(
                          decoded, StubOperandRole::kSource1,
                          StubFragmentKind::kMatrix) == 1 &&
                      CountDescriptorsForRoleAndFragmentKind(
                          decoded, StubOperandRole::kAccumulator,
                          StubFragmentKind::kMatrix) == 1 &&
                      CountDescriptorsForRoleAndComponentCount(
                          decoded, StubOperandRole::kDestination, 1) == 1 &&
                      CountDescriptorsForRoleAndComponentCount(
                          decoded, StubOperandRole::kSource0, 1) == 1 &&
                      CountDescriptorsForRoleAndComponentCount(
                          decoded, StubOperandRole::kSource1, 1) == 1 &&
                      CountDescriptorsForRoleAndComponentCount(
                          decoded, StubOperandRole::kAccumulator, 1) == 1 &&
                      CountSlotsOfKindAndValueClass(
                          decoded, StubOperandSlotKind::kDestination,
                          StubOperandValueClass::kMatrixFragment) == 1 &&
                      CountSlotsOfKindAndValueClass(
                          decoded, StubOperandSlotKind::kSource0,
                          StubOperandValueClass::kMatrixFragment) == 1 &&
                      CountSlotsOfKindAndValueClass(
                          decoded, StubOperandSlotKind::kSource1,
                          StubOperandValueClass::kMatrixFragment) == 1 &&
                      CountSlotsOfKindAndValueClass(
                          decoded, StubOperandSlotKind::kAccumulatorSource,
                          StubOperandValueClass::kAccumulatorFragment) == 1 &&
                      CountSlotsOfKindAndFragmentKind(
                          decoded, StubOperandSlotKind::kDestination,
                          StubFragmentKind::kMatrix) == 1 &&
                      CountSlotsOfKindAndFragmentKind(
                          decoded, StubOperandSlotKind::kSource0,
                          StubFragmentKind::kMatrix) == 1 &&
                      CountSlotsOfKindAndFragmentKind(
                          decoded, StubOperandSlotKind::kSource1,
                          StubFragmentKind::kMatrix) == 1 &&
                      CountSlotsOfKindAndFragmentKind(
                          decoded, StubOperandSlotKind::kAccumulatorSource,
                          StubFragmentKind::kMatrix) == 1 &&
                      CountSlotsOfKindAndComponentCount(
                          decoded, StubOperandSlotKind::kDestination, 1) == 1 &&
                      CountSlotsOfKindAndComponentCount(
                          decoded, StubOperandSlotKind::kSource0, 1) == 1 &&
                      CountSlotsOfKindAndComponentCount(
                          decoded, StubOperandSlotKind::kSource1, 1) == 1 &&
                      CountSlotsOfKindAndComponentCount(
                          decoded, StubOperandSlotKind::kAccumulatorSource, 1) == 1 &&
                      CountSlotsOfKindWithOutputFlag(
                          decoded, StubOperandSlotKind::kDestination, true) == 1 &&
                      CountSlotsOfKindWithOutputFlag(
                          decoded, StubOperandSlotKind::kSource0, false) == 1 &&
                      CountSlotsOfKindWithOutputFlag(
                          decoded, StubOperandSlotKind::kSource1, false) == 1 &&
                      CountSlotsOfKindWithOutputFlag(
                          decoded, StubOperandSlotKind::kAccumulatorSource,
                          false) == 1 &&
                      CountSlotsWithFragmentKindAndWaveSize(
                          decoded, StubFragmentKind::kMatrix, 32) == 4 &&
                      CountDescriptorsWithFragmentKindAndWaveSize(
                          decoded, StubFragmentKind::kMatrix, 32) == 4 &&
                      CountSlotsWithFragmentKindAndWaveSize(
                          decoded, StubFragmentKind::kScalar, 0) == 0 &&
                      CountDescriptorsWithFragmentKindAndWaveSize(
                          decoded, StubFragmentKind::kScalar, 0) == 0 &&
                      CountSlotsWithValueClass(
                          decoded, StubOperandValueClass::kMatrixFragment) == 3 &&
                      CountSlotsWithValueClass(
                          decoded,
                          StubOperandValueClass::kAccumulatorFragment) == 1 &&
                      CountSlotsWithValueClassAndComponentCount(
                          decoded, StubOperandValueClass::kMatrixFragment, 1) == 3 &&
                      CountSlotsWithValueClassAndComponentCount(
                          decoded,
                          StubOperandValueClass::kAccumulatorFragment, 1) == 1 &&
                      CountDescriptorsWithValueClass(
                          decoded, StubOperandValueClass::kMatrixFragment) == 3 &&
                      CountDescriptorsWithValueClass(
                          decoded,
                          StubOperandValueClass::kAccumulatorFragment) == 1 &&
                      CountDescriptorsWithValueClassAndComponentCount(
                          decoded, StubOperandValueClass::kMatrixFragment, 1) == 3 &&
                      CountDescriptorsWithValueClassAndComponentCount(
                          decoded,
                          StubOperandValueClass::kAccumulatorFragment, 1) == 1,
                  "expected routed WMMA/SWMMAC core seed to keep exact matrix composition")) {
        return 1;
      }
      if (!Expect(
              CountDescriptorsForRoleAndSlotKind(
                  decoded, StubOperandRole::kDestination,
                  StubOperandSlotKind::kDestination) == 1 &&
                  CountDescriptorsForRoleAndSlotKind(
                      decoded, StubOperandRole::kSource0,
                      StubOperandSlotKind::kSource0) == 1 &&
                  CountDescriptorsForRoleAndSlotKind(
                      decoded, StubOperandRole::kSource1,
                      StubOperandSlotKind::kSource1) == 1 &&
                  CountDescriptorsForRoleAndSlotKind(
                      decoded, StubOperandRole::kAccumulator,
                      StubOperandSlotKind::kAccumulatorSource) == 1 &&
                  CountSlotsOfKindAndLogicalOperandIndex(
                      decoded, StubOperandSlotKind::kDestination, 0) == 1 &&
                  CountSlotsOfKindAndLogicalOperandIndex(
                      decoded, StubOperandSlotKind::kSource0, 1) == 1 &&
                  CountSlotsOfKindAndLogicalOperandIndex(
                      decoded, StubOperandSlotKind::kSource1, 2) == 1 &&
                  CountSlotsOfKindAndLogicalOperandIndex(
                      decoded, StubOperandSlotKind::kAccumulatorSource, 3) ==
                      1,
              "expected routed WMMA/SWMMAC core seed to keep exact role/slot and logical-index mapping")) {
        return 1;
      }
      if (!Expect(
              CountDescriptorsForRoleAndWaveSize(
                  decoded, StubOperandRole::kDestination, 32) == 1 &&
                  CountDescriptorsForRoleAndWaveSize(
                      decoded, StubOperandRole::kSource0, 32) == 1 &&
                  CountDescriptorsForRoleAndWaveSize(
                      decoded, StubOperandRole::kSource1, 32) == 1 &&
                  CountDescriptorsForRoleAndWaveSize(
                      decoded, StubOperandRole::kAccumulator, 32) == 1 &&
                  CountSlotsOfKindAndWaveSize(
                      decoded, StubOperandSlotKind::kDestination, 32) == 1 &&
                  CountSlotsOfKindAndWaveSize(
                      decoded, StubOperandSlotKind::kSource0, 32) == 1 &&
                  CountSlotsOfKindAndWaveSize(
                      decoded, StubOperandSlotKind::kSource1, 32) == 1 &&
                  CountSlotsOfKindAndWaveSize(
                      decoded, StubOperandSlotKind::kAccumulatorSource, 32) ==
                      1,
              "expected routed WMMA/SWMMAC core seed to keep exact role/slot wave-size mapping")) {
        return 1;
      }
      const std::uint8_t core_destination_element_bit_width =
          FindUniqueSlotElementBitWidth(decoded, StubOperandSlotKind::kDestination);
      const std::uint8_t core_source0_element_bit_width =
          FindUniqueSlotElementBitWidth(decoded, StubOperandSlotKind::kSource0);
      const std::uint8_t core_source1_element_bit_width =
          FindUniqueSlotElementBitWidth(decoded, StubOperandSlotKind::kSource1);
      const std::uint8_t core_accumulator_element_bit_width =
          FindUniqueSlotElementBitWidth(decoded, StubOperandSlotKind::kAccumulatorSource);
      if (!Expect(
              core_destination_element_bit_width != 0xff &&
                  core_source0_element_bit_width != 0xff &&
                  core_source1_element_bit_width != 0xff &&
                  core_accumulator_element_bit_width != 0xff &&
                  core_source0_element_bit_width ==
                      core_source1_element_bit_width &&
                  core_destination_element_bit_width ==
                      core_accumulator_element_bit_width &&
                  CountDescriptorsForRoleAndElementBitWidth(
                      decoded, StubOperandRole::kDestination,
                      core_destination_element_bit_width) == 1 &&
                  CountDescriptorsForRoleAndElementBitWidth(
                      decoded, StubOperandRole::kSource0,
                      core_source0_element_bit_width) == 1 &&
                  CountDescriptorsForRoleAndElementBitWidth(
                      decoded, StubOperandRole::kSource1,
                      core_source1_element_bit_width) == 1 &&
                  CountDescriptorsForRoleAndElementBitWidth(
                      decoded, StubOperandRole::kAccumulator,
                      core_accumulator_element_bit_width) == 1 &&
                  CountSlotsOfKindAndElementBitWidth(
                      decoded, StubOperandSlotKind::kDestination,
                      core_destination_element_bit_width) == 1 &&
                  CountSlotsOfKindAndElementBitWidth(
                      decoded, StubOperandSlotKind::kSource0,
                      core_source0_element_bit_width) == 1 &&
                  CountSlotsOfKindAndElementBitWidth(
                      decoded, StubOperandSlotKind::kSource1,
                      core_source1_element_bit_width) == 1 &&
                  CountSlotsOfKindAndElementBitWidth(
                      decoded, StubOperandSlotKind::kAccumulatorSource,
                      core_accumulator_element_bit_width) == 1,
              "expected routed WMMA/SWMMAC core seed to keep exact role/slot element-width mapping")) {
        return 1;
      }
      if (!Expect(
              CountDescriptorsForRoleAndPackedElements(
                  decoded, StubOperandRole::kDestination, 0) == 1 &&
                  CountDescriptorsForRoleAndPackedElements(
                      decoded, StubOperandRole::kSource0, 0) == 1 &&
                  CountDescriptorsForRoleAndPackedElements(
                      decoded, StubOperandRole::kSource1, 0) == 1 &&
                  CountDescriptorsForRoleAndPackedElements(
                      decoded, StubOperandRole::kAccumulator, 0) == 1 &&
                  CountSlotsOfKindAndPackedElements(
                      decoded, StubOperandSlotKind::kDestination, 0) == 1 &&
                  CountSlotsOfKindAndPackedElements(
                      decoded, StubOperandSlotKind::kSource0, 0) == 1 &&
                  CountSlotsOfKindAndPackedElements(
                      decoded, StubOperandSlotKind::kSource1, 0) == 1 &&
                  CountSlotsOfKindAndPackedElements(
                      decoded, StubOperandSlotKind::kAccumulatorSource, 0) ==
                      1,
              "expected routed WMMA/SWMMAC core seed to keep exact role/slot packed-elements mapping")) {
        return 1;
      }
      const ShapeExtents core_destination_extents =
          FindUniqueSlotShapeExtents(decoded, StubOperandSlotKind::kDestination);
      const ShapeExtents core_source0_extents =
          FindUniqueSlotShapeExtents(decoded, StubOperandSlotKind::kSource0);
      const ShapeExtents core_source1_extents =
          FindUniqueSlotShapeExtents(decoded, StubOperandSlotKind::kSource1);
      const ShapeExtents core_accumulator_extents =
          FindUniqueSlotShapeExtents(decoded, StubOperandSlotKind::kAccumulatorSource);
      if (!Expect(
              core_destination_extents.valid && core_source0_extents.valid &&
                  core_source1_extents.valid && core_accumulator_extents.valid &&
                  core_destination_extents.rows == core_source0_extents.rows &&
                  core_destination_extents.columns ==
                      core_source0_extents.columns &&
                  core_destination_extents.depth ==
                      core_source0_extents.depth &&
                  core_destination_extents.rows == core_source1_extents.rows &&
                  core_destination_extents.columns ==
                      core_source1_extents.columns &&
                  core_destination_extents.depth ==
                      core_source1_extents.depth &&
                  core_destination_extents.rows ==
                      core_accumulator_extents.rows &&
                  core_destination_extents.columns ==
                      core_accumulator_extents.columns &&
                  core_destination_extents.depth ==
                      core_accumulator_extents.depth &&
                  CountDescriptorsForRoleAndDimensions(
                      decoded, StubOperandRole::kDestination,
                      core_destination_extents.rows,
                      core_destination_extents.columns,
                      core_destination_extents.depth) == 1 &&
                  CountDescriptorsForRoleAndDimensions(
                      decoded, StubOperandRole::kSource0,
                      core_source0_extents.rows, core_source0_extents.columns,
                      core_source0_extents.depth) == 1 &&
                  CountDescriptorsForRoleAndDimensions(
                      decoded, StubOperandRole::kSource1,
                      core_source1_extents.rows, core_source1_extents.columns,
                      core_source1_extents.depth) == 1 &&
                  CountDescriptorsForRoleAndDimensions(
                      decoded, StubOperandRole::kAccumulator,
                      core_accumulator_extents.rows,
                      core_accumulator_extents.columns,
                      core_accumulator_extents.depth) == 1 &&
                  CountSlotsOfKindAndDimensions(
                      decoded, StubOperandSlotKind::kDestination,
                      core_destination_extents.rows,
                      core_destination_extents.columns,
                      core_destination_extents.depth) == 1 &&
                  CountSlotsOfKindAndDimensions(
                      decoded, StubOperandSlotKind::kSource0,
                      core_source0_extents.rows, core_source0_extents.columns,
                      core_source0_extents.depth) == 1 &&
                  CountSlotsOfKindAndDimensions(
                      decoded, StubOperandSlotKind::kSource1,
                      core_source1_extents.rows, core_source1_extents.columns,
                      core_source1_extents.depth) == 1 &&
                  CountSlotsOfKindAndDimensions(
                      decoded, StubOperandSlotKind::kAccumulatorSource,
                      core_accumulator_extents.rows,
                      core_accumulator_extents.columns,
                      core_accumulator_extents.depth) == 1,
              "expected routed WMMA/SWMMAC core seed to keep exact role/slot dimension mapping")) {
        return 1;
      }
      if (!Expect(
              CountRoleBindings(decoded, StubOperandRole::kDestination) == 1 &&
                  CountRoleBindings(decoded, StubOperandRole::kSource0) == 1 &&
                  CountRoleBindings(decoded, StubOperandRole::kSource1) == 1 &&
                  CountRoleBindings(decoded, StubOperandRole::kAccumulator) == 1 &&
                  CountRoleBindings(decoded, StubOperandRole::kScale) == 0 &&
                  CountRoleBindingsWithCount(decoded, StubOperandRole::kDestination,
                                             1) == 1 &&
                  CountRoleBindingsWithCount(decoded, StubOperandRole::kSource0,
                                             1) == 1 &&
                  CountRoleBindingsWithCount(decoded, StubOperandRole::kSource1,
                                             1) == 1 &&
                  CountRoleBindingsWithCount(decoded,
                                             StubOperandRole::kAccumulator,
                                             1) == 1 &&
                  CountRoleBindingsWithOutputFlag(
                      decoded, StubOperandRole::kDestination, true) == 1 &&
                  CountRoleBindingsWithOutputFlag(
                      decoded, StubOperandRole::kSource0, false) == 1 &&
                  CountRoleBindingsWithOutputFlag(
                      decoded, StubOperandRole::kSource1, false) == 1 &&
                  CountRoleBindingsWithOutputFlag(
                      decoded, StubOperandRole::kAccumulator, false) == 1,
              "expected routed WMMA/SWMMAC core seed to keep exact operand-role binding mapping")) {
        return 1;
      }
      if (!Expect(decoded.operand_roles.binding_count == 4 &&
                      AllRoleBindingsExplicit(decoded),
                  "expected routed WMMA/SWMMAC core seed to keep exact operand-role binding-count and explicitness")) {
        return 1;
      }
      if (!Expect(
              MatchesRoleBindingSequence(
                  decoded,
                  {{StubOperandRole::kSource0, 1, false, false},
                   {StubOperandRole::kSource1, 1, false, false},
                   {StubOperandRole::kAccumulator, 1, false, false},
                   {StubOperandRole::kDestination, 1, true, false}}) &&
                  MatchesSlotBindingSequence(
                      decoded,
                      {{StubOperandSlotKind::kDestination,
                        StubOperandValueClass::kMatrixFragment,
                        0,
                        1,
                        true,
                        false},
                       {StubOperandSlotKind::kSource0,
                        StubOperandValueClass::kMatrixFragment,
                        1,
                        1,
                        false,
                        false},
                       {StubOperandSlotKind::kSource1,
                        StubOperandValueClass::kMatrixFragment,
                        2,
                        1,
                        false,
                        false},
                       {StubOperandSlotKind::kAccumulatorSource,
                        StubOperandValueClass::kAccumulatorFragment,
                        3,
                        1,
                        false,
                        false}}) &&
                  MatchesDescriptorSequence(
                      decoded,
                      {{StubOperandRole::kDestination,
                        StubOperandSlotKind::kDestination,
                        StubOperandValueClass::kMatrixFragment,
                        StubOperandAccess::kWrite,
                        1,
                        false},
                       {StubOperandRole::kSource0,
                        StubOperandSlotKind::kSource0,
                        StubOperandValueClass::kMatrixFragment,
                        StubOperandAccess::kRead,
                        1,
                        false},
                       {StubOperandRole::kSource1,
                        StubOperandSlotKind::kSource1,
                        StubOperandValueClass::kMatrixFragment,
                        StubOperandAccess::kRead,
                        1,
                        false},
                       {StubOperandRole::kAccumulator,
                        StubOperandSlotKind::kAccumulatorSource,
                        StubOperandValueClass::kAccumulatorFragment,
                        StubOperandAccess::kRead,
                        1,
                        false}}),
              "expected routed WMMA/SWMMAC core seed to keep exact operand-role, slot, and descriptor order")) {
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
    const StubDecoderRouteInfo* route_info =
        FindStubDecoderRouteInfo(instruction_name);
    if (!Expect(decoded.status == StubDecodeStatus::kDecodedStub,
                "expected routed tensor seed to decode")) {
      return 1;
    }
    if (!Expect(route_info != nullptr &&
                    MatchesRouteInfoPayload(decoded, *route_info),
                "expected routed tensor seed to preserve route-info and RDNA4 provenance")) {
      return 1;
    }
    if (!Expect(
            MatchesRouteMetadata(decoded,
                                 {StubDecoderRoute::kMimgTensor,
                                  "kMimgTensor",
                                  "DecodeMimgTensorStub",
                                  2}),
            "expected routed tensor seed to keep exact route metadata")) {
      return 1;
    }
    if (!Expect(
            decoded.execution_domain == StubExecutionDomain::kTensorMemory &&
                decoded.opcode_shape ==
                    (instruction_name == "TENSOR_LOAD_TO_LDS"
                         ? StubOpcodeShape::kTensorLoadToLds
                         : StubOpcodeShape::kTensorStoreFromLds) &&
                MatchesLayout(
                    decoded,
                    {instruction_name == "TENSOR_LOAD_TO_LDS"
                         ? StubOperandLayoutKind::kTensorLoadToLds
                         : StubOperandLayoutKind::kTensorStoreFromLds,
                     2,
                     0,
                     0,
                     false,
                     false,
                     true,
                     true,
                     instruction_name == "TENSOR_STORE_FROM_LDS"}),
            "expected routed tensor seed to keep exact top-level route/layout metadata")) {
      return 1;
    }
    if (!Expect(MatchesTopLevelFlags(decoded, false, true, false, false),
                "expected routed tensor seed to keep exact top-level flag composition")) {
      return 1;
    }
    if (!Expect(MatchesLayoutToRecordInvariants(decoded),
                "expected routed tensor seed to keep exact layout-to-record consistency")) {
      return 1;
    }
    if (!Expect(MatchesDescriptorToSlotParity(decoded),
                "expected routed tensor seed to keep exact descriptor-to-slot parity")) {
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
                    CountSlotsWithValueClass(
                        decoded, StubOperandValueClass::kTensorDescriptor) == 1 &&
                    CountSlotsWithValueClass(
                        decoded, StubOperandValueClass::kTensorCoordinate) == 1 &&
                    CountSlotsWithValueClass(
                        decoded, StubOperandValueClass::kLdsAddress) == 1 &&
                    CountSlotsWithValueClassAndComponentCount(
                        decoded, StubOperandValueClass::kTensorDescriptor, 1) == 1 &&
                    CountSlotsWithValueClassAndComponentCount(
                        decoded, StubOperandValueClass::kTensorCoordinate, 1) == 1 &&
                    CountSlotsWithValueClassAndComponentCount(
                        decoded, StubOperandValueClass::kLdsAddress, 1) == 1 &&
                    CountDescriptorsWithValueClass(
                        decoded, StubOperandValueClass::kTensorDescriptor) == 1 &&
                    CountDescriptorsWithValueClass(
                        decoded, StubOperandValueClass::kTensorCoordinate) == 1 &&
                    CountDescriptorsWithValueClass(
                        decoded, StubOperandValueClass::kLdsAddress) == 1 &&
                    CountDescriptorsWithValueClassAndComponentCount(
                        decoded, StubOperandValueClass::kTensorDescriptor, 1) == 1 &&
                    CountDescriptorsWithValueClassAndComponentCount(
                        decoded, StubOperandValueClass::kTensorCoordinate, 1) == 1 &&
                    CountDescriptorsWithValueClassAndComponentCount(
                        decoded, StubOperandValueClass::kLdsAddress, 1) == 1 &&
                    CountOutputSlots(decoded) ==
                        (instruction_name == "TENSOR_LOAD_TO_LDS" ? 1u : 0u) &&
                    CountDescriptorsWithAccess(decoded, StubOperandAccess::kRead) ==
                        (instruction_name == "TENSOR_LOAD_TO_LDS" ? 2u : 3u) &&
                    CountDescriptorsWithAccess(decoded, StubOperandAccess::kWrite) ==
                        (instruction_name == "TENSOR_LOAD_TO_LDS" ? 1u : 0u) &&
                    CountDescriptorsForRoleAndAccess(
                        decoded, StubOperandRole::kTensorDescriptor,
                        StubOperandAccess::kRead) == 1 &&
                    CountDescriptorsForRoleAndAccess(
                        decoded, StubOperandRole::kTensorCoordinate,
                        StubOperandAccess::kRead) == 1 &&
                    CountDescriptorsForRoleAndAccess(
                        decoded,
                        instruction_name == "TENSOR_LOAD_TO_LDS"
                            ? StubOperandRole::kLdsDestination
                            : StubOperandRole::kLdsSource,
                        instruction_name == "TENSOR_LOAD_TO_LDS"
                            ? StubOperandAccess::kWrite
                            : StubOperandAccess::kRead) == 1 &&
                    CountDescriptorsForRoleAndValueClass(
                        decoded, StubOperandRole::kTensorDescriptor,
                        StubOperandValueClass::kTensorDescriptor) == 1 &&
                    CountDescriptorsForRoleAndValueClass(
                        decoded, StubOperandRole::kTensorCoordinate,
                        StubOperandValueClass::kTensorCoordinate) == 1 &&
                    CountDescriptorsForRoleAndValueClass(
                        decoded,
                        instruction_name == "TENSOR_LOAD_TO_LDS"
                            ? StubOperandRole::kLdsDestination
                            : StubOperandRole::kLdsSource,
                        StubOperandValueClass::kLdsAddress) == 1 &&
                    CountDescriptorsForRoleAndFragmentKind(
                        decoded, StubOperandRole::kTensorDescriptor,
                        StubFragmentKind::kTensorDescriptor) == 1 &&
                    CountDescriptorsForRoleAndFragmentKind(
                        decoded, StubOperandRole::kTensorCoordinate,
                        StubFragmentKind::kTensorCoordinate) == 1 &&
                    CountDescriptorsForRoleAndFragmentKind(
                        decoded,
                        instruction_name == "TENSOR_LOAD_TO_LDS"
                            ? StubOperandRole::kLdsDestination
                            : StubOperandRole::kLdsSource,
                        StubFragmentKind::kAddress) == 1 &&
                    CountDescriptorsForRoleAndComponentCount(
                        decoded, StubOperandRole::kTensorDescriptor, 1) == 1 &&
                    CountDescriptorsForRoleAndComponentCount(
                        decoded, StubOperandRole::kTensorCoordinate, 1) == 1 &&
                    CountDescriptorsForRoleAndComponentCount(
                        decoded,
                        instruction_name == "TENSOR_LOAD_TO_LDS"
                            ? StubOperandRole::kLdsDestination
                            : StubOperandRole::kLdsSource,
                        1) == 1 &&
                    CountSlotsOfKindAndValueClass(
                        decoded, StubOperandSlotKind::kTensorDescriptorSource,
                        StubOperandValueClass::kTensorDescriptor) == 1 &&
                    CountSlotsOfKindAndValueClass(
                        decoded, StubOperandSlotKind::kTensorCoordinateSource,
                        StubOperandValueClass::kTensorCoordinate) == 1 &&
                    CountSlotsOfKindAndValueClass(
                        decoded,
                        instruction_name == "TENSOR_LOAD_TO_LDS"
                            ? StubOperandSlotKind::kLdsDestination
                            : StubOperandSlotKind::kLdsSource,
                        StubOperandValueClass::kLdsAddress) == 1 &&
                    CountSlotsOfKindAndFragmentKind(
                        decoded, StubOperandSlotKind::kTensorDescriptorSource,
                        StubFragmentKind::kTensorDescriptor) == 1 &&
                    CountSlotsOfKindAndFragmentKind(
                        decoded, StubOperandSlotKind::kTensorCoordinateSource,
                        StubFragmentKind::kTensorCoordinate) == 1 &&
                    CountSlotsOfKindAndFragmentKind(
                        decoded,
                        instruction_name == "TENSOR_LOAD_TO_LDS"
                            ? StubOperandSlotKind::kLdsDestination
                            : StubOperandSlotKind::kLdsSource,
                        StubFragmentKind::kAddress) == 1 &&
                    CountSlotsOfKindAndComponentCount(
                        decoded, StubOperandSlotKind::kTensorDescriptorSource,
                        1) == 1 &&
                    CountSlotsOfKindAndComponentCount(
                        decoded, StubOperandSlotKind::kTensorCoordinateSource,
                        1) == 1 &&
                    CountSlotsOfKindAndComponentCount(
                        decoded,
                        instruction_name == "TENSOR_LOAD_TO_LDS"
                            ? StubOperandSlotKind::kLdsDestination
                            : StubOperandSlotKind::kLdsSource,
                        1) == 1 &&
                    CountSlotsOfKindWithOutputFlag(
                        decoded, StubOperandSlotKind::kTensorDescriptorSource,
                        false) == 1 &&
                    CountSlotsOfKindWithOutputFlag(
                        decoded, StubOperandSlotKind::kTensorCoordinateSource,
                        false) == 1 &&
                    CountSlotsOfKindWithOutputFlag(
                        decoded,
                        instruction_name == "TENSOR_LOAD_TO_LDS"
                            ? StubOperandSlotKind::kLdsDestination
                            : StubOperandSlotKind::kLdsSource,
                        instruction_name == "TENSOR_LOAD_TO_LDS") == 1 &&
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
    if (!Expect(
            CountDescriptorsForRoleAndSlotKind(
                decoded, StubOperandRole::kTensorDescriptor,
                StubOperandSlotKind::kTensorDescriptorSource) == 1 &&
                CountDescriptorsForRoleAndSlotKind(
                    decoded, StubOperandRole::kTensorCoordinate,
                    StubOperandSlotKind::kTensorCoordinateSource) == 1 &&
                CountDescriptorsForRoleAndSlotKind(
                    decoded,
                    instruction_name == "TENSOR_LOAD_TO_LDS"
                        ? StubOperandRole::kLdsDestination
                        : StubOperandRole::kLdsSource,
                    instruction_name == "TENSOR_LOAD_TO_LDS"
                        ? StubOperandSlotKind::kLdsDestination
                        : StubOperandSlotKind::kLdsSource) == 1 &&
                CountSlotsOfKindAndLogicalOperandIndex(
                    decoded, StubOperandSlotKind::kTensorDescriptorSource, 0) ==
                    1 &&
                CountSlotsOfKindAndLogicalOperandIndex(
                    decoded, StubOperandSlotKind::kTensorCoordinateSource, 1) ==
                    1 &&
                CountSlotsOfKindAndLogicalOperandIndex(
                    decoded,
                    instruction_name == "TENSOR_LOAD_TO_LDS"
                        ? StubOperandSlotKind::kLdsDestination
                        : StubOperandSlotKind::kLdsSource,
                    2) == 1,
            "expected routed tensor seed to keep exact role/slot and logical-index mapping")) {
      return 1;
    }
    if (!Expect(
            CountDescriptorsForRoleAndWaveSize(
                decoded, StubOperandRole::kTensorDescriptor, 0) == 1 &&
                CountDescriptorsForRoleAndWaveSize(
                    decoded, StubOperandRole::kTensorCoordinate, 0) == 1 &&
                CountDescriptorsForRoleAndWaveSize(
                    decoded,
                    instruction_name == "TENSOR_LOAD_TO_LDS"
                        ? StubOperandRole::kLdsDestination
                        : StubOperandRole::kLdsSource,
                    0) == 1 &&
                CountSlotsOfKindAndWaveSize(
                    decoded, StubOperandSlotKind::kTensorDescriptorSource, 0) ==
                    1 &&
                CountSlotsOfKindAndWaveSize(
                    decoded, StubOperandSlotKind::kTensorCoordinateSource, 0) ==
                    1 &&
                CountSlotsOfKindAndWaveSize(
                    decoded,
                    instruction_name == "TENSOR_LOAD_TO_LDS"
                        ? StubOperandSlotKind::kLdsDestination
                        : StubOperandSlotKind::kLdsSource,
                    0) == 1,
            "expected routed tensor seed to keep exact role/slot wave-size mapping")) {
      return 1;
    }
    if (!Expect(
            CountDescriptorsForRoleAndElementBitWidth(
                decoded, StubOperandRole::kTensorDescriptor, 0) == 1 &&
                CountDescriptorsForRoleAndElementBitWidth(
                    decoded, StubOperandRole::kTensorCoordinate, 0) == 1 &&
                CountDescriptorsForRoleAndElementBitWidth(
                    decoded,
                    instruction_name == "TENSOR_LOAD_TO_LDS"
                        ? StubOperandRole::kLdsDestination
                        : StubOperandRole::kLdsSource,
                    32) == 1 &&
                CountSlotsOfKindAndElementBitWidth(
                    decoded, StubOperandSlotKind::kTensorDescriptorSource, 0) ==
                    1 &&
                CountSlotsOfKindAndElementBitWidth(
                    decoded, StubOperandSlotKind::kTensorCoordinateSource, 0) ==
                    1 &&
                CountSlotsOfKindAndElementBitWidth(
                    decoded,
                    instruction_name == "TENSOR_LOAD_TO_LDS"
                        ? StubOperandSlotKind::kLdsDestination
                        : StubOperandSlotKind::kLdsSource,
                    32) == 1,
            "expected routed tensor seed to keep exact role/slot element-width mapping")) {
      return 1;
    }
    if (!Expect(
            CountDescriptorsForRoleAndPackedElements(
                decoded, StubOperandRole::kTensorDescriptor, 1) == 1 &&
                CountDescriptorsForRoleAndPackedElements(
                    decoded, StubOperandRole::kTensorCoordinate, 1) == 1 &&
                CountDescriptorsForRoleAndPackedElements(
                    decoded,
                    instruction_name == "TENSOR_LOAD_TO_LDS"
                        ? StubOperandRole::kLdsDestination
                        : StubOperandRole::kLdsSource,
                    1) == 1 &&
                CountSlotsOfKindAndPackedElements(
                    decoded, StubOperandSlotKind::kTensorDescriptorSource, 1) ==
                    1 &&
                CountSlotsOfKindAndPackedElements(
                    decoded, StubOperandSlotKind::kTensorCoordinateSource, 1) ==
                    1 &&
                CountSlotsOfKindAndPackedElements(
                    decoded,
                    instruction_name == "TENSOR_LOAD_TO_LDS"
                        ? StubOperandSlotKind::kLdsDestination
                        : StubOperandSlotKind::kLdsSource,
                    1) == 1,
            "expected routed tensor seed to keep exact role/slot packed-elements mapping")) {
      return 1;
    }
    if (!Expect(
            CountDescriptorsForRoleAndDimensions(
                decoded, StubOperandRole::kTensorDescriptor, 1, 1, 1) == 1 &&
                CountDescriptorsForRoleAndDimensions(
                    decoded, StubOperandRole::kTensorCoordinate, 1, 1, 1) == 1 &&
                CountDescriptorsForRoleAndDimensions(
                    decoded,
                    instruction_name == "TENSOR_LOAD_TO_LDS"
                        ? StubOperandRole::kLdsDestination
                        : StubOperandRole::kLdsSource,
                    1, 1, 1) == 1 &&
                CountSlotsOfKindAndDimensions(
                    decoded, StubOperandSlotKind::kTensorDescriptorSource, 1, 1,
                    1) == 1 &&
                CountSlotsOfKindAndDimensions(
                    decoded, StubOperandSlotKind::kTensorCoordinateSource, 1, 1,
                    1) == 1 &&
                CountSlotsOfKindAndDimensions(
                    decoded,
                    instruction_name == "TENSOR_LOAD_TO_LDS"
                        ? StubOperandSlotKind::kLdsDestination
                        : StubOperandSlotKind::kLdsSource,
                    1, 1, 1) == 1,
            "expected routed tensor seed to keep exact role/slot dimension mapping")) {
      return 1;
    }
    if (!Expect(
            CountRoleBindings(decoded, StubOperandRole::kTensorDescriptor) == 1 &&
                CountRoleBindings(decoded, StubOperandRole::kTensorCoordinate) ==
                    1 &&
                CountRoleBindings(
                    decoded,
                    instruction_name == "TENSOR_LOAD_TO_LDS"
                        ? StubOperandRole::kLdsDestination
                        : StubOperandRole::kLdsSource) == 1 &&
                CountRoleBindingsWithCount(
                    decoded, StubOperandRole::kTensorDescriptor, 1) == 1 &&
                CountRoleBindingsWithCount(
                    decoded, StubOperandRole::kTensorCoordinate, 1) == 1 &&
                CountRoleBindingsWithCount(
                    decoded,
                    instruction_name == "TENSOR_LOAD_TO_LDS"
                        ? StubOperandRole::kLdsDestination
                        : StubOperandRole::kLdsSource,
                    1) == 1 &&
                CountRoleBindingsWithOutputFlag(
                    decoded, StubOperandRole::kTensorDescriptor, false) == 1 &&
                CountRoleBindingsWithOutputFlag(
                    decoded, StubOperandRole::kTensorCoordinate, false) == 1 &&
                CountRoleBindingsWithOutputFlag(
                    decoded,
                    instruction_name == "TENSOR_LOAD_TO_LDS"
                        ? StubOperandRole::kLdsDestination
                        : StubOperandRole::kLdsSource,
                    instruction_name == "TENSOR_LOAD_TO_LDS") == 1,
            "expected routed tensor seed to keep exact operand-role binding mapping")) {
      return 1;
    }
    if (!Expect(decoded.operand_roles.binding_count == 3 &&
                    AllRoleBindingsExplicit(decoded),
                "expected routed tensor seed to keep exact operand-role binding-count and explicitness")) {
      return 1;
    }
    if (instruction_name == "TENSOR_LOAD_TO_LDS") {
      if (!Expect(
              MatchesRoleBindingSequence(
                  decoded,
                  {{StubOperandRole::kTensorDescriptor, 1, false, false},
                   {StubOperandRole::kTensorCoordinate, 1, false, false},
                   {StubOperandRole::kLdsDestination, 1, true, false}}) &&
                  MatchesSlotBindingSequence(
                      decoded,
                      {{StubOperandSlotKind::kTensorDescriptorSource,
                        StubOperandValueClass::kTensorDescriptor,
                        0,
                        1,
                        false,
                        false},
                       {StubOperandSlotKind::kTensorCoordinateSource,
                        StubOperandValueClass::kTensorCoordinate,
                        1,
                        1,
                        false,
                        false},
                       {StubOperandSlotKind::kLdsDestination,
                        StubOperandValueClass::kLdsAddress,
                        2,
                        1,
                        true,
                        false}}) &&
                  MatchesDescriptorSequence(
                      decoded,
                      {{StubOperandRole::kTensorDescriptor,
                        StubOperandSlotKind::kTensorDescriptorSource,
                        StubOperandValueClass::kTensorDescriptor,
                        StubOperandAccess::kRead,
                        1,
                        false},
                       {StubOperandRole::kTensorCoordinate,
                        StubOperandSlotKind::kTensorCoordinateSource,
                        StubOperandValueClass::kTensorCoordinate,
                        StubOperandAccess::kRead,
                        1,
                        false},
                       {StubOperandRole::kLdsDestination,
                        StubOperandSlotKind::kLdsDestination,
                        StubOperandValueClass::kLdsAddress,
                        StubOperandAccess::kWrite,
                        1,
                        false}}),
              "expected routed tensor-load seed to keep exact operand-role, slot, and descriptor order")) {
        return 1;
      }
    } else {
      if (!Expect(
              MatchesRoleBindingSequence(
                  decoded,
                  {{StubOperandRole::kTensorDescriptor, 1, false, false},
                   {StubOperandRole::kTensorCoordinate, 1, false, false},
                   {StubOperandRole::kLdsSource, 1, false, false}}) &&
                  MatchesSlotBindingSequence(
                      decoded,
                      {{StubOperandSlotKind::kTensorDescriptorSource,
                        StubOperandValueClass::kTensorDescriptor,
                        0,
                        1,
                        false,
                        false},
                       {StubOperandSlotKind::kTensorCoordinateSource,
                        StubOperandValueClass::kTensorCoordinate,
                        1,
                        1,
                        false,
                        false},
                       {StubOperandSlotKind::kLdsSource,
                        StubOperandValueClass::kLdsAddress,
                        2,
                        1,
                        false,
                        false}}) &&
                  MatchesDescriptorSequence(
                      decoded,
                      {{StubOperandRole::kTensorDescriptor,
                        StubOperandSlotKind::kTensorDescriptorSource,
                        StubOperandValueClass::kTensorDescriptor,
                        StubOperandAccess::kRead,
                        1,
                        false},
                       {StubOperandRole::kTensorCoordinate,
                        StubOperandSlotKind::kTensorCoordinateSource,
                        StubOperandValueClass::kTensorCoordinate,
                        StubOperandAccess::kRead,
                        1,
                        false},
                       {StubOperandRole::kLdsSource,
                        StubOperandSlotKind::kLdsSource,
                        StubOperandValueClass::kLdsAddress,
                        StubOperandAccess::kRead,
                        1,
                        false}}),
              "expected routed tensor-store seed to keep exact operand-role, slot, and descriptor order")) {
        return 1;
      }
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
    const StubDecoderRouteInfo* route_info =
        FindStubDecoderRouteInfo(instruction_name);
    if (!Expect(decoded.status == StubDecodeStatus::kDecodedStub,
                "expected routed VOP1 seed to decode")) {
      return 1;
    }
    if (!Expect(route_info != nullptr &&
                    MatchesRouteInfoPayload(decoded, *route_info),
                "expected routed VOP1 seed to preserve route-info and RDNA4 provenance")) {
      return 1;
    }
    if (!Expect(
            MatchesRouteMetadata(
                decoded,
                {StubDecoderRoute::kVop1, "kVop1", "DecodeVop1Stub", 3}),
            "expected routed VOP1 seed to keep exact route metadata")) {
      return 1;
    }
    if (!Expect(
            decoded.execution_domain == StubExecutionDomain::kConversion &&
                decoded.opcode_shape ==
                    (instruction_name == "V_CVT_F16_FP8" ||
                             instruction_name == "V_CVT_F16_BF8"
                         ? StubOpcodeShape::kFp8ConvertToF16
                         : instruction_name == "V_CVT_F32_FP8"
                               ? StubOpcodeShape::kFp8ConvertToF32
                               : StubOpcodeShape::kFp8PackedConvert) &&
                MatchesLayout(
                    decoded,
                    {instruction_name == "V_CVT_F16_FP8"
                         ? StubOperandLayoutKind::kCvtF16Fp8
                         : instruction_name == "V_CVT_F16_BF8"
                               ? StubOperandLayoutKind::kCvtF16Bf8
                               : instruction_name == "V_CVT_F32_FP8"
                                     ? StubOperandLayoutKind::kCvtF32Fp8
                                     : instruction_name == "V_CVT_PK_F16_FP8"
                                           ? StubOperandLayoutKind::kCvtPkF16Fp8
                                           : StubOperandLayoutKind::kCvtPkF16Bf8,
                     1,
                     1,
                     0,
                     false,
                     false,
                                     false,
                                     false,
                                     false}),
            "expected routed VOP1 seed to keep exact top-level route/layout metadata")) {
      return 1;
    }
    if (!Expect(
            MatchesTopLevelFlags(decoded, false, false, false,
                                 instruction_name.find("PK_") !=
                                     std::string_view::npos),
            "expected routed VOP1 seed to keep exact top-level flag composition")) {
      return 1;
    }
    if (!Expect(MatchesLayoutToRecordInvariants(decoded),
                "expected routed VOP1 seed to keep exact layout-to-record consistency")) {
      return 1;
    }
    if (!Expect(MatchesDescriptorToSlotParity(decoded),
                "expected routed VOP1 seed to keep exact descriptor-to-slot parity")) {
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
                    CountSlotsWithValueClass(
                        decoded,
                        instruction_name.find("PK_") != std::string_view::npos
                            ? StubOperandValueClass::kPackedVector
                            : StubOperandValueClass::kVectorRegister) == 2 &&
                    CountDescriptorsWithValueClass(
                        decoded,
                        instruction_name.find("PK_") != std::string_view::npos
                            ? StubOperandValueClass::kPackedVector
                            : StubOperandValueClass::kVectorRegister) == 2 &&
                    CountSlotsWithValueClassAndComponentCount(
                        decoded,
                        instruction_name.find("PK_") != std::string_view::npos
                            ? StubOperandValueClass::kPackedVector
                            : StubOperandValueClass::kVectorRegister,
                        instruction_name.find("PK_") != std::string_view::npos ? 2 : 1) ==
                        2 &&
                    CountDescriptorsWithValueClassAndComponentCount(
                        decoded,
                        instruction_name.find("PK_") != std::string_view::npos
                            ? StubOperandValueClass::kPackedVector
                            : StubOperandValueClass::kVectorRegister,
                        instruction_name.find("PK_") != std::string_view::npos ? 2 : 1) ==
                        2 &&
                    CountOutputSlots(decoded) == 1 &&
                    CountDescriptorsWithAccess(decoded, StubOperandAccess::kRead) == 1 &&
                    CountDescriptorsWithAccess(decoded, StubOperandAccess::kWrite) == 1 &&
                    CountDescriptorsForRoleAndAccess(
                        decoded, StubOperandRole::kSource0,
                        StubOperandAccess::kRead) == 1 &&
                    CountDescriptorsForRoleAndAccess(
                        decoded, StubOperandRole::kDestination,
                        StubOperandAccess::kWrite) == 1 &&
                    CountDescriptorsForRoleAndValueClass(
                        decoded, StubOperandRole::kSource0,
                        instruction_name.find("PK_") != std::string_view::npos
                            ? StubOperandValueClass::kPackedVector
                            : StubOperandValueClass::kVectorRegister) == 1 &&
                    CountDescriptorsForRoleAndValueClass(
                        decoded, StubOperandRole::kDestination,
                        instruction_name.find("PK_") != std::string_view::npos
                            ? StubOperandValueClass::kPackedVector
                            : StubOperandValueClass::kVectorRegister) == 1 &&
                    CountDescriptorsForRoleAndFragmentKind(
                        decoded, StubOperandRole::kSource0,
                        instruction_name.find("PK_") != std::string_view::npos
                            ? StubFragmentKind::kPacked
                            : StubFragmentKind::kScalar) == 1 &&
                    CountDescriptorsForRoleAndFragmentKind(
                        decoded, StubOperandRole::kDestination,
                        instruction_name.find("PK_") != std::string_view::npos
                            ? StubFragmentKind::kPacked
                            : StubFragmentKind::kScalar) == 1 &&
                    CountDescriptorsForRoleAndComponentCount(
                        decoded, StubOperandRole::kSource0,
                        instruction_name.find("PK_") != std::string_view::npos ? 2 : 1) ==
                        1 &&
                    CountDescriptorsForRoleAndComponentCount(
                        decoded, StubOperandRole::kDestination,
                        instruction_name.find("PK_") != std::string_view::npos ? 2 : 1) ==
                        1 &&
                    CountSlotsOfKindAndValueClass(
                        decoded, StubOperandSlotKind::kSource0,
                        instruction_name.find("PK_") != std::string_view::npos
                            ? StubOperandValueClass::kPackedVector
                            : StubOperandValueClass::kVectorRegister) == 1 &&
                    CountSlotsOfKindAndValueClass(
                        decoded, StubOperandSlotKind::kDestination,
                        instruction_name.find("PK_") != std::string_view::npos
                            ? StubOperandValueClass::kPackedVector
                            : StubOperandValueClass::kVectorRegister) == 1 &&
                    CountSlotsOfKindAndFragmentKind(
                        decoded, StubOperandSlotKind::kSource0,
                        instruction_name.find("PK_") != std::string_view::npos
                            ? StubFragmentKind::kPacked
                            : StubFragmentKind::kScalar) == 1 &&
                    CountSlotsOfKindAndFragmentKind(
                        decoded, StubOperandSlotKind::kDestination,
                        instruction_name.find("PK_") != std::string_view::npos
                            ? StubFragmentKind::kPacked
                            : StubFragmentKind::kScalar) == 1 &&
                    CountSlotsOfKindAndComponentCount(
                        decoded, StubOperandSlotKind::kSource0,
                        instruction_name.find("PK_") != std::string_view::npos ? 2 : 1) ==
                        1 &&
                    CountSlotsOfKindAndComponentCount(
                        decoded, StubOperandSlotKind::kDestination,
                        instruction_name.find("PK_") != std::string_view::npos ? 2 : 1) ==
                        1 &&
                    CountSlotsOfKindWithOutputFlag(
                        decoded, StubOperandSlotKind::kSource0, false) == 1 &&
                    CountSlotsOfKindWithOutputFlag(
                        decoded, StubOperandSlotKind::kDestination, true) == 1 &&
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
    if (!Expect(
            CountDescriptorsForRoleAndSlotKind(
                decoded, StubOperandRole::kSource0,
                StubOperandSlotKind::kSource0) == 1 &&
                CountDescriptorsForRoleAndSlotKind(
                    decoded, StubOperandRole::kDestination,
                    StubOperandSlotKind::kDestination) == 1 &&
                CountSlotsOfKindAndLogicalOperandIndex(
                    decoded, StubOperandSlotKind::kDestination, 0) == 1 &&
                CountSlotsOfKindAndLogicalOperandIndex(
                    decoded, StubOperandSlotKind::kSource0, 1) == 1,
            "expected routed VOP1 seed to keep exact role/slot and logical-index mapping")) {
      return 1;
    }
    if (!Expect(
            CountDescriptorsForRoleAndWaveSize(
                decoded, StubOperandRole::kSource0, 0) == 1 &&
                CountDescriptorsForRoleAndWaveSize(
                    decoded, StubOperandRole::kDestination, 0) == 1 &&
                CountSlotsOfKindAndWaveSize(
                    decoded, StubOperandSlotKind::kSource0, 0) == 1 &&
                CountSlotsOfKindAndWaveSize(
                    decoded, StubOperandSlotKind::kDestination, 0) == 1,
            "expected routed VOP1 seed to keep exact role/slot wave-size mapping")) {
      return 1;
    }
    if (!Expect(
            CountDescriptorsForRoleAndElementBitWidth(
                decoded, StubOperandRole::kSource0, 8) == 1 &&
                CountDescriptorsForRoleAndElementBitWidth(
                    decoded, StubOperandRole::kDestination,
                    instruction_name == "V_CVT_F32_FP8"
                        ? 32
                        : 16) == 1 &&
                CountSlotsOfKindAndElementBitWidth(
                    decoded, StubOperandSlotKind::kSource0, 8) == 1 &&
                CountSlotsOfKindAndElementBitWidth(
                    decoded, StubOperandSlotKind::kDestination,
                    instruction_name == "V_CVT_F32_FP8"
                        ? 32
                        : 16) == 1,
            "expected routed VOP1 seed to keep exact role/slot element-width mapping")) {
      return 1;
    }
    if (!Expect(
            CountDescriptorsForRoleAndPackedElements(
                decoded, StubOperandRole::kSource0,
                instruction_name.find("PK_") != std::string_view::npos ? 2 : 1) ==
                1 &&
                CountDescriptorsForRoleAndPackedElements(
                    decoded, StubOperandRole::kDestination,
                    instruction_name.find("PK_") != std::string_view::npos ? 2
                                                                           : 1) ==
                    1 &&
                CountSlotsOfKindAndPackedElements(
                    decoded, StubOperandSlotKind::kSource0,
                    instruction_name.find("PK_") != std::string_view::npos ? 2 : 1) ==
                    1 &&
                CountSlotsOfKindAndPackedElements(
                    decoded, StubOperandSlotKind::kDestination,
                    instruction_name.find("PK_") != std::string_view::npos ? 2
                                                                           : 1) ==
                    1,
            "expected routed VOP1 seed to keep exact role/slot packed-elements mapping")) {
      return 1;
    }
    if (!Expect(
            CountDescriptorsForRoleAndDimensions(
                decoded, StubOperandRole::kSource0, 1, 1, 1) == 1 &&
                CountDescriptorsForRoleAndDimensions(
                    decoded, StubOperandRole::kDestination, 1, 1, 1) == 1 &&
                CountSlotsOfKindAndDimensions(
                    decoded, StubOperandSlotKind::kSource0, 1, 1, 1) == 1 &&
                CountSlotsOfKindAndDimensions(
                    decoded, StubOperandSlotKind::kDestination, 1, 1, 1) == 1,
            "expected routed VOP1 seed to keep exact role/slot dimension mapping")) {
      return 1;
    }
    if (!Expect(
            CountRoleBindings(decoded, StubOperandRole::kSource0) == 1 &&
                CountRoleBindings(decoded, StubOperandRole::kDestination) == 1 &&
                CountRoleBindingsWithCount(decoded, StubOperandRole::kSource0,
                                           1) == 1 &&
                CountRoleBindingsWithCount(
                    decoded, StubOperandRole::kDestination, 1) == 1 &&
                CountRoleBindingsWithOutputFlag(
                    decoded, StubOperandRole::kSource0, false) == 1 &&
                CountRoleBindingsWithOutputFlag(
                    decoded, StubOperandRole::kDestination, true) == 1,
            "expected routed VOP1 seed to keep exact operand-role binding mapping")) {
      return 1;
    }
    if (!Expect(decoded.operand_roles.binding_count == 2 &&
                    AllRoleBindingsExplicit(decoded),
                "expected routed VOP1 seed to keep exact operand-role binding-count and explicitness")) {
      return 1;
    }
    if (!Expect(
            MatchesRoleBindingSequence(
                decoded,
                {{StubOperandRole::kSource0, 1, false, false},
                 {StubOperandRole::kDestination, 1, true, false}}) &&
                MatchesSlotBindingSequence(
                    decoded,
                    {{StubOperandSlotKind::kDestination,
                      instruction_name.find("PK_") != std::string_view::npos
                          ? StubOperandValueClass::kPackedVector
                          : StubOperandValueClass::kVectorRegister,
                      0,
                      instruction_name.find("PK_") != std::string_view::npos ? 2u
                                                                              : 1u,
                      true,
                      false},
                     {StubOperandSlotKind::kSource0,
                      instruction_name.find("PK_") != std::string_view::npos
                          ? StubOperandValueClass::kPackedVector
                          : StubOperandValueClass::kVectorRegister,
                      1,
                      instruction_name.find("PK_") != std::string_view::npos ? 2u
                                                                              : 1u,
                      false,
                      false}}) &&
                MatchesDescriptorSequence(
                    decoded,
                    {{StubOperandRole::kDestination,
                      StubOperandSlotKind::kDestination,
                      instruction_name.find("PK_") != std::string_view::npos
                          ? StubOperandValueClass::kPackedVector
                          : StubOperandValueClass::kVectorRegister,
                      StubOperandAccess::kWrite,
                      static_cast<std::uint8_t>(
                          instruction_name.find("PK_") != std::string_view::npos
                              ? 2u
                              : 1u),
                      false},
                     {StubOperandRole::kSource0,
                      StubOperandSlotKind::kSource0,
                      instruction_name.find("PK_") != std::string_view::npos
                          ? StubOperandValueClass::kPackedVector
                          : StubOperandValueClass::kVectorRegister,
                      StubOperandAccess::kRead,
                      static_cast<std::uint8_t>(
                          instruction_name.find("PK_") != std::string_view::npos
                              ? 2u
                              : 1u),
                      false}}),
            "expected routed VOP1 seed to keep exact operand-role, slot, and descriptor order")) {
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
      if (!Expect(
              CountDescriptorsForRoleAndElementBitWidth(
                  decoded, StubOperandRole::kSource0, 8) == 1 &&
                  CountDescriptorsForRoleAndElementBitWidth(
                      decoded, StubOperandRole::kDestination, 16) == 1 &&
                  CountSlotsOfKindAndElementBitWidth(
                      decoded, StubOperandSlotKind::kSource0, 8) == 1 &&
                  CountSlotsOfKindAndElementBitWidth(
                      decoded, StubOperandSlotKind::kDestination, 16) == 1,
              "expected routed packed VOP1 seed to keep exact role/slot element-width mapping")) {
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
    const StubDecoderRouteInfo* route_info =
        FindStubDecoderRouteInfo(instruction_name);
    if (!Expect(decoded.status == StubDecodeStatus::kDecodedStub,
                "expected routed VOP3 SDST seed to decode")) {
      return 1;
    }
    if (!Expect(route_info != nullptr &&
                    MatchesRouteInfoPayload(decoded, *route_info),
                "expected routed VOP3 SDST seed to preserve route-info and RDNA4 provenance")) {
      return 1;
    }
    if (!Expect(
            MatchesRouteMetadata(decoded,
                                 {StubDecoderRoute::kVop3Sdst,
                                  "kVop3Sdst",
                                  "DecodeVop3SdstStub",
                                  4}),
            "expected routed VOP3 SDST seed to keep exact route metadata")) {
      return 1;
    }
    if (!Expect(
            decoded.execution_domain == StubExecutionDomain::kScaleAssist &&
                decoded.opcode_shape == StubOpcodeShape::kVop3SdstScale &&
                MatchesLayout(decoded,
                              {StubOperandLayoutKind::kVDivScaleF64,
                               3,
                               2,
                               0,
                               true,
                               false,
                               false,
                               false,
                               false}),
            "expected routed VOP3 SDST seed to keep exact top-level route/layout metadata")) {
      return 1;
    }
    if (!Expect(MatchesTopLevelFlags(decoded, false, false, true, false),
                "expected routed VOP3 SDST seed to keep exact top-level flag composition")) {
      return 1;
    }
    if (!Expect(MatchesLayoutToRecordInvariants(decoded),
                "expected routed VOP3 SDST seed to keep exact layout-to-record consistency")) {
      return 1;
    }
    if (!Expect(MatchesDescriptorToSlotParity(decoded),
                "expected routed VOP3 SDST seed to keep exact descriptor-to-slot parity")) {
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
                    CountSlotsWithValueClass(
                        decoded, StubOperandValueClass::kVectorRegister) == 4 &&
                    CountSlotsWithValueClass(
                        decoded, StubOperandValueClass::kScalarRegister) == 1 &&
                    CountSlotsWithValueClassAndComponentCount(
                        decoded, StubOperandValueClass::kVectorRegister, 2) == 4 &&
                    CountSlotsWithValueClassAndComponentCount(
                        decoded, StubOperandValueClass::kScalarRegister, 1) == 1 &&
                    CountDescriptorsWithValueClass(
                        decoded, StubOperandValueClass::kVectorRegister) == 4 &&
                    CountDescriptorsWithValueClass(
                        decoded, StubOperandValueClass::kScalarRegister) == 1 &&
                    CountDescriptorsWithValueClassAndComponentCount(
                        decoded, StubOperandValueClass::kVectorRegister, 2) == 4 &&
                    CountDescriptorsWithValueClassAndComponentCount(
                        decoded, StubOperandValueClass::kScalarRegister, 1) == 1 &&
                    CountOutputSlots(decoded) == 2 &&
                    CountDescriptorsWithAccess(decoded, StubOperandAccess::kRead) == 3 &&
                    CountDescriptorsWithAccess(decoded, StubOperandAccess::kWrite) == 2 &&
                    CountDescriptorsForRoleAndAccess(
                        decoded, StubOperandRole::kSource0,
                        StubOperandAccess::kRead) == 1 &&
                    CountDescriptorsForRoleAndAccess(
                        decoded, StubOperandRole::kSource1,
                        StubOperandAccess::kRead) == 1 &&
                    CountDescriptorsForRoleAndAccess(
                        decoded, StubOperandRole::kScale,
                        StubOperandAccess::kRead) == 1 &&
                    CountDescriptorsForRoleAndAccess(
                        decoded, StubOperandRole::kDestination,
                        StubOperandAccess::kWrite) == 2 &&
                    CountDescriptorsForRoleAndValueClass(
                        decoded, StubOperandRole::kSource0,
                        StubOperandValueClass::kVectorRegister) == 1 &&
                    CountDescriptorsForRoleAndValueClass(
                        decoded, StubOperandRole::kSource1,
                        StubOperandValueClass::kVectorRegister) == 1 &&
                    CountDescriptorsForRoleAndValueClass(
                        decoded, StubOperandRole::kScale,
                        StubOperandValueClass::kVectorRegister) == 1 &&
                    CountDescriptorsForRoleAndValueClass(
                        decoded, StubOperandRole::kDestination,
                        StubOperandValueClass::kVectorRegister) == 1 &&
                    CountDescriptorsForRoleAndValueClass(
                        decoded, StubOperandRole::kDestination,
                        StubOperandValueClass::kScalarRegister) == 1 &&
                    CountDescriptorsForRoleAndFragmentKind(
                        decoded, StubOperandRole::kSource0,
                        StubFragmentKind::kScalar) == 1 &&
                    CountDescriptorsForRoleAndFragmentKind(
                        decoded, StubOperandRole::kSource1,
                        StubFragmentKind::kScalar) == 1 &&
                    CountDescriptorsForRoleAndFragmentKind(
                        decoded, StubOperandRole::kScale,
                        StubFragmentKind::kScalar) == 1 &&
                    CountDescriptorsForRoleAndFragmentKind(
                        decoded, StubOperandRole::kDestination,
                        StubFragmentKind::kScalar) == 2 &&
                    CountDescriptorsForRoleAndComponentCount(
                        decoded, StubOperandRole::kSource0, 2) == 1 &&
                    CountDescriptorsForRoleAndComponentCount(
                        decoded, StubOperandRole::kSource1, 2) == 1 &&
                    CountDescriptorsForRoleAndComponentCount(
                        decoded, StubOperandRole::kScale, 2) == 1 &&
                    CountDescriptorsForRoleAndComponentCount(
                        decoded, StubOperandRole::kDestination, 2) == 1 &&
                    CountDescriptorsForRoleAndComponentCount(
                        decoded, StubOperandRole::kDestination, 1) == 1 &&
                    CountSlotsOfKindAndValueClass(
                        decoded, StubOperandSlotKind::kDestination,
                        StubOperandValueClass::kVectorRegister) == 1 &&
                    CountSlotsOfKindAndValueClass(
                        decoded, StubOperandSlotKind::kScalarDestination,
                        StubOperandValueClass::kScalarRegister) == 1 &&
                    CountSlotsOfKindAndValueClass(
                        decoded, StubOperandSlotKind::kSource0,
                        StubOperandValueClass::kVectorRegister) == 1 &&
                    CountSlotsOfKindAndValueClass(
                        decoded, StubOperandSlotKind::kSource1,
                        StubOperandValueClass::kVectorRegister) == 1 &&
                    CountSlotsOfKindAndValueClass(
                        decoded, StubOperandSlotKind::kScaleSource,
                        StubOperandValueClass::kVectorRegister) == 1 &&
                    CountSlotsOfKindAndFragmentKind(
                        decoded, StubOperandSlotKind::kDestination,
                        StubFragmentKind::kScalar) == 1 &&
                    CountSlotsOfKindAndFragmentKind(
                        decoded, StubOperandSlotKind::kScalarDestination,
                        StubFragmentKind::kScalar) == 1 &&
                    CountSlotsOfKindAndFragmentKind(
                        decoded, StubOperandSlotKind::kSource0,
                        StubFragmentKind::kScalar) == 1 &&
                    CountSlotsOfKindAndFragmentKind(
                        decoded, StubOperandSlotKind::kSource1,
                        StubFragmentKind::kScalar) == 1 &&
                    CountSlotsOfKindAndFragmentKind(
                        decoded, StubOperandSlotKind::kScaleSource,
                        StubFragmentKind::kScalar) == 1 &&
                    CountSlotsOfKindAndComponentCount(
                        decoded, StubOperandSlotKind::kDestination, 2) == 1 &&
                    CountSlotsOfKindAndComponentCount(
                        decoded, StubOperandSlotKind::kScalarDestination, 1) == 1 &&
                    CountSlotsOfKindAndComponentCount(
                        decoded, StubOperandSlotKind::kSource0, 2) == 1 &&
                    CountSlotsOfKindAndComponentCount(
                        decoded, StubOperandSlotKind::kSource1, 2) == 1 &&
                    CountSlotsOfKindAndComponentCount(
                        decoded, StubOperandSlotKind::kScaleSource, 2) == 1 &&
                    CountSlotsOfKindWithOutputFlag(
                        decoded, StubOperandSlotKind::kDestination, true) == 1 &&
                    CountSlotsOfKindWithOutputFlag(
                        decoded, StubOperandSlotKind::kScalarDestination,
                        true) == 1 &&
                    CountSlotsOfKindWithOutputFlag(
                        decoded, StubOperandSlotKind::kSource0, false) == 1 &&
                    CountSlotsOfKindWithOutputFlag(
                        decoded, StubOperandSlotKind::kSource1, false) == 1 &&
                    CountSlotsOfKindWithOutputFlag(
                        decoded, StubOperandSlotKind::kScaleSource, false) == 1 &&
                    !decoded.uses_tensor_memory &&
                    AllSlotsExplicit(decoded) &&
                    AllDescriptorsExplicit(decoded) &&
                    AllSlotWaveSizesAre(decoded, 0) &&
                    AllDescriptorWaveSizesAre(decoded, 0),
                "expected routed VOP3 SDST seed to keep non-matrix wave-size semantics")) {
      return 1;
    }
    if (!Expect(
            CountDescriptorsForRoleAndSlotKind(
                decoded, StubOperandRole::kSource0,
                StubOperandSlotKind::kSource0) == 1 &&
                CountDescriptorsForRoleAndSlotKind(
                    decoded, StubOperandRole::kSource1,
                    StubOperandSlotKind::kSource1) == 1 &&
                CountDescriptorsForRoleAndSlotKind(
                    decoded, StubOperandRole::kScale,
                    StubOperandSlotKind::kScaleSource) == 1 &&
                CountDescriptorsForRoleAndSlotKind(
                    decoded, StubOperandRole::kDestination,
                    StubOperandSlotKind::kDestination) == 1 &&
                CountDescriptorsForRoleAndSlotKind(
                    decoded, StubOperandRole::kDestination,
                    StubOperandSlotKind::kScalarDestination) == 1 &&
                CountSlotsOfKindAndLogicalOperandIndex(
                    decoded, StubOperandSlotKind::kDestination, 0) == 1 &&
                CountSlotsOfKindAndLogicalOperandIndex(
                    decoded, StubOperandSlotKind::kScalarDestination, 1) == 1 &&
                CountSlotsOfKindAndLogicalOperandIndex(
                    decoded, StubOperandSlotKind::kSource0, 2) == 1 &&
                CountSlotsOfKindAndLogicalOperandIndex(
                    decoded, StubOperandSlotKind::kSource1, 3) == 1 &&
                CountSlotsOfKindAndLogicalOperandIndex(
                    decoded, StubOperandSlotKind::kScaleSource, 4) == 1,
            "expected routed VOP3 SDST seed to keep exact role/slot and logical-index mapping")) {
      return 1;
    }
    if (!Expect(
            CountDescriptorsForRoleAndWaveSize(
                decoded, StubOperandRole::kSource0, 0) == 1 &&
                CountDescriptorsForRoleAndWaveSize(
                    decoded, StubOperandRole::kSource1, 0) == 1 &&
                CountDescriptorsForRoleAndWaveSize(
                    decoded, StubOperandRole::kScale, 0) == 1 &&
                CountDescriptorsForRoleAndWaveSize(
                    decoded, StubOperandRole::kDestination, 0) == 2 &&
                CountSlotsOfKindAndWaveSize(
                    decoded, StubOperandSlotKind::kDestination, 0) == 1 &&
                CountSlotsOfKindAndWaveSize(
                    decoded, StubOperandSlotKind::kScalarDestination, 0) == 1 &&
                CountSlotsOfKindAndWaveSize(
                    decoded, StubOperandSlotKind::kSource0, 0) == 1 &&
                CountSlotsOfKindAndWaveSize(
                    decoded, StubOperandSlotKind::kSource1, 0) == 1 &&
                CountSlotsOfKindAndWaveSize(
                    decoded, StubOperandSlotKind::kScaleSource, 0) == 1,
            "expected routed VOP3 SDST seed to keep exact role/slot wave-size mapping")) {
      return 1;
    }
    if (!Expect(
            CountDescriptorsForRoleAndElementBitWidth(
                decoded, StubOperandRole::kSource0, 64) == 1 &&
                CountDescriptorsForRoleAndElementBitWidth(
                    decoded, StubOperandRole::kSource1, 64) == 1 &&
                CountDescriptorsForRoleAndElementBitWidth(
                    decoded, StubOperandRole::kScale, 64) == 1 &&
                CountDescriptorsForRoleAndElementBitWidth(
                    decoded, StubOperandRole::kDestination, 64) == 1 &&
                CountDescriptorsForRoleAndElementBitWidth(
                    decoded, StubOperandRole::kDestination, 32) == 1 &&
                CountSlotsOfKindAndElementBitWidth(
                    decoded, StubOperandSlotKind::kSource0, 64) == 1 &&
                CountSlotsOfKindAndElementBitWidth(
                    decoded, StubOperandSlotKind::kSource1, 64) == 1 &&
                CountSlotsOfKindAndElementBitWidth(
                    decoded, StubOperandSlotKind::kScaleSource, 64) == 1 &&
                CountSlotsOfKindAndElementBitWidth(
                    decoded, StubOperandSlotKind::kDestination, 64) == 1 &&
                CountSlotsOfKindAndElementBitWidth(
                    decoded, StubOperandSlotKind::kScalarDestination, 32) == 1,
            "expected routed VOP3 SDST seed to keep exact role/slot element-width mapping")) {
      return 1;
    }
    if (!Expect(
            CountDescriptorsForRoleAndPackedElements(
                decoded, StubOperandRole::kSource0, 1) == 1 &&
                CountDescriptorsForRoleAndPackedElements(
                    decoded, StubOperandRole::kSource1, 1) == 1 &&
                CountDescriptorsForRoleAndPackedElements(
                    decoded, StubOperandRole::kScale, 1) == 1 &&
                CountDescriptorsForRoleAndPackedElements(
                    decoded, StubOperandRole::kDestination, 1) == 2 &&
                CountSlotsOfKindAndPackedElements(
                    decoded, StubOperandSlotKind::kSource0, 1) == 1 &&
                CountSlotsOfKindAndPackedElements(
                    decoded, StubOperandSlotKind::kSource1, 1) == 1 &&
                CountSlotsOfKindAndPackedElements(
                    decoded, StubOperandSlotKind::kScaleSource, 1) == 1 &&
                CountSlotsOfKindAndPackedElements(
                    decoded, StubOperandSlotKind::kDestination, 1) == 1 &&
                CountSlotsOfKindAndPackedElements(
                    decoded, StubOperandSlotKind::kScalarDestination, 1) == 1,
            "expected routed VOP3 SDST seed to keep exact role/slot packed-elements mapping")) {
      return 1;
    }
    if (!Expect(
            CountDescriptorsForRoleAndDimensions(
                decoded, StubOperandRole::kSource0, 1, 1, 1) == 1 &&
                CountDescriptorsForRoleAndDimensions(
                    decoded, StubOperandRole::kSource1, 1, 1, 1) == 1 &&
                CountDescriptorsForRoleAndDimensions(
                    decoded, StubOperandRole::kScale, 1, 1, 1) == 1 &&
                CountDescriptorsForRoleAndDimensions(
                    decoded, StubOperandRole::kDestination, 1, 1, 1) == 2 &&
                CountSlotsOfKindAndDimensions(
                    decoded, StubOperandSlotKind::kSource0, 1, 1, 1) == 1 &&
                CountSlotsOfKindAndDimensions(
                    decoded, StubOperandSlotKind::kSource1, 1, 1, 1) == 1 &&
                CountSlotsOfKindAndDimensions(
                    decoded, StubOperandSlotKind::kScaleSource, 1, 1, 1) == 1 &&
                CountSlotsOfKindAndDimensions(
                    decoded, StubOperandSlotKind::kDestination, 1, 1, 1) == 1 &&
                CountSlotsOfKindAndDimensions(
                    decoded, StubOperandSlotKind::kScalarDestination, 1, 1, 1) ==
                    1,
            "expected routed VOP3 SDST seed to keep exact role/slot dimension mapping")) {
      return 1;
    }
    if (!Expect(
            CountRoleBindings(decoded, StubOperandRole::kSource0) == 1 &&
                CountRoleBindings(decoded, StubOperandRole::kSource1) == 1 &&
                CountRoleBindings(decoded, StubOperandRole::kScale) == 1 &&
                CountRoleBindings(decoded, StubOperandRole::kDestination) == 1 &&
                CountRoleBindingsWithCount(decoded, StubOperandRole::kSource0,
                                           1) == 1 &&
                CountRoleBindingsWithCount(decoded, StubOperandRole::kSource1,
                                           1) == 1 &&
                CountRoleBindingsWithCount(decoded, StubOperandRole::kScale,
                                           1) == 1 &&
                CountRoleBindingsWithCount(
                    decoded, StubOperandRole::kDestination, 1) == 1 &&
                CountRoleBindingsWithOutputFlag(
                    decoded, StubOperandRole::kSource0, false) == 1 &&
                CountRoleBindingsWithOutputFlag(
                    decoded, StubOperandRole::kSource1, false) == 1 &&
                CountRoleBindingsWithOutputFlag(
                    decoded, StubOperandRole::kScale, false) == 1 &&
                CountRoleBindingsWithOutputFlag(
                    decoded, StubOperandRole::kDestination, true) == 1,
            "expected routed VOP3 SDST seed to keep exact operand-role binding mapping")) {
      return 1;
    }
    if (!Expect(decoded.operand_roles.binding_count == 4 &&
                    AllRoleBindingsExplicit(decoded),
                "expected routed VOP3 SDST seed to keep exact operand-role binding-count and explicitness")) {
      return 1;
    }
    if (!Expect(
            MatchesRoleBindingSequence(
                decoded,
                {{StubOperandRole::kSource0, 1, false, false},
                 {StubOperandRole::kSource1, 1, false, false},
                 {StubOperandRole::kScale, 1, false, false},
                 {StubOperandRole::kDestination, 1, true, false}}) &&
                MatchesSlotBindingSequence(
                    decoded,
                    {{StubOperandSlotKind::kDestination,
                      StubOperandValueClass::kVectorRegister,
                      0,
                      2,
                      true,
                      false},
                     {StubOperandSlotKind::kScalarDestination,
                      StubOperandValueClass::kScalarRegister,
                      1,
                      1,
                      true,
                      false},
                     {StubOperandSlotKind::kSource0,
                      StubOperandValueClass::kVectorRegister,
                      2,
                      2,
                      false,
                      false},
                     {StubOperandSlotKind::kSource1,
                      StubOperandValueClass::kVectorRegister,
                      3,
                      2,
                      false,
                      false},
                     {StubOperandSlotKind::kScaleSource,
                      StubOperandValueClass::kVectorRegister,
                      4,
                      2,
                      false,
                      false}}) &&
                MatchesDescriptorSequence(
                    decoded,
                    {{StubOperandRole::kDestination,
                      StubOperandSlotKind::kDestination,
                      StubOperandValueClass::kVectorRegister,
                      StubOperandAccess::kWrite,
                      2,
                      false},
                     {StubOperandRole::kDestination,
                      StubOperandSlotKind::kScalarDestination,
                      StubOperandValueClass::kScalarRegister,
                      StubOperandAccess::kWrite,
                      1,
                      false},
                     {StubOperandRole::kSource0,
                      StubOperandSlotKind::kSource0,
                      StubOperandValueClass::kVectorRegister,
                      StubOperandAccess::kRead,
                      2,
                      false},
                     {StubOperandRole::kSource1,
                      StubOperandSlotKind::kSource1,
                      StubOperandValueClass::kVectorRegister,
                      StubOperandAccess::kRead,
                      2,
                      false},
                     {StubOperandRole::kScale,
                      StubOperandSlotKind::kScaleSource,
                      StubOperandValueClass::kVectorRegister,
                      StubOperandAccess::kRead,
                      2,
                      false}}),
            "expected routed VOP3 SDST seed to keep exact operand-role, slot, and descriptor order")) {
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
    const StubDecoderRouteInfo* route_info =
        FindStubDecoderRouteInfo(instruction_name);
    if (!Expect(decoded.status == StubDecodeStatus::kDecodedStub,
                "expected routed paired-scale seed to decode")) {
      return 1;
    }
    if (!Expect(route_info != nullptr &&
                    MatchesRouteInfoPayload(decoded, *route_info),
                "expected routed paired-scale seed to preserve route-info and RDNA4 provenance")) {
      return 1;
    }
    if (!Expect(
            MatchesRouteMetadata(
                decoded,
                {StubDecoderRoute::kVop3p, "kVop3p", "DecodeVop3pStub", 1}),
            "expected routed paired-scale seed to keep exact route metadata")) {
      return 1;
    }
    if (!Expect(
            decoded.execution_domain == StubExecutionDomain::kMatrix &&
                decoded.opcode_shape == StubOpcodeShape::kWmmaScalePairedLoad &&
                MatchesLayout(
                    decoded,
                    {instruction_name == "V_WMMA_LD_SCALE_PAIRED_B32"
                         ? StubOperandLayoutKind::kWmmaLdScalePairedB32
                         : StubOperandLayoutKind::kWmmaLdScale16PairedB64,
                     2,
                     1,
                     0,
                     true,
                     true,
                     false,
                     false,
                     false}),
            "expected routed paired-scale seed to keep exact top-level route/layout metadata")) {
      return 1;
    }
    if (!Expect(MatchesTopLevelFlags(decoded, false, false, true, true),
                "expected routed paired-scale seed to keep exact top-level flag composition")) {
      return 1;
    }
    if (!Expect(MatchesLayoutToRecordInvariants(decoded),
                "expected routed paired-scale seed to keep exact layout-to-record consistency")) {
      return 1;
    }
    if (!Expect(MatchesDescriptorToSlotParity(decoded),
                "expected routed paired-scale seed to keep exact descriptor-to-slot parity")) {
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
                    CountSlotsWithValueClass(
                        decoded, StubOperandValueClass::kVectorRegister) == 2 &&
                    CountSlotsWithValueClass(
                        decoded, StubOperandValueClass::kScalarRegister) == 2 &&
                    CountSlotsWithValueClassAndComponentCount(
                        decoded, StubOperandValueClass::kVectorRegister, 1) == 2 &&
                    CountSlotsWithValueClassAndComponentCount(
                        decoded, StubOperandValueClass::kScalarRegister, 1) == 2 &&
                    CountDescriptorsWithValueClass(
                        decoded, StubOperandValueClass::kVectorRegister) == 2 &&
                    CountDescriptorsWithValueClass(
                        decoded, StubOperandValueClass::kScalarRegister) == 2 &&
                    CountDescriptorsWithValueClassAndComponentCount(
                        decoded, StubOperandValueClass::kVectorRegister, 1) == 2 &&
                    CountDescriptorsWithValueClassAndComponentCount(
                        decoded, StubOperandValueClass::kScalarRegister, 1) == 2 &&
                    CountOutputSlots(decoded) == 1 &&
                    CountDescriptorsWithAccess(decoded, StubOperandAccess::kRead) == 3 &&
                    CountDescriptorsWithAccess(decoded, StubOperandAccess::kWrite) == 1 &&
                    CountDescriptorsForRoleAndAccess(
                        decoded, StubOperandRole::kSource0,
                        StubOperandAccess::kRead) == 1 &&
                    CountDescriptorsForRoleAndAccess(
                        decoded, StubOperandRole::kScale,
                        StubOperandAccess::kRead) == 1 &&
                    CountDescriptorsForRoleAndAccess(
                        decoded, StubOperandRole::kPairedScale,
                        StubOperandAccess::kRead) == 1 &&
                    CountDescriptorsForRoleAndAccess(
                        decoded, StubOperandRole::kDestination,
                        StubOperandAccess::kWrite) == 1 &&
                    CountDescriptorsForRoleAndValueClass(
                        decoded, StubOperandRole::kSource0,
                        StubOperandValueClass::kVectorRegister) == 1 &&
                    CountDescriptorsForRoleAndValueClass(
                        decoded, StubOperandRole::kScale,
                        StubOperandValueClass::kScalarRegister) == 1 &&
                    CountDescriptorsForRoleAndValueClass(
                        decoded, StubOperandRole::kPairedScale,
                        StubOperandValueClass::kScalarRegister) == 1 &&
                    CountDescriptorsForRoleAndValueClass(
                        decoded, StubOperandRole::kDestination,
                        StubOperandValueClass::kVectorRegister) == 1 &&
                    CountDescriptorsForRoleAndFragmentKind(
                        decoded, StubOperandRole::kSource0,
                        StubFragmentKind::kVector) == 1 &&
                    CountDescriptorsForRoleAndFragmentKind(
                        decoded, StubOperandRole::kScale,
                        StubFragmentKind::kScalar) == 1 &&
                    CountDescriptorsForRoleAndFragmentKind(
                        decoded, StubOperandRole::kPairedScale,
                        StubFragmentKind::kScalar) == 1 &&
                    CountDescriptorsForRoleAndFragmentKind(
                        decoded, StubOperandRole::kDestination,
                        StubFragmentKind::kVector) == 1 &&
                    CountDescriptorsForRoleAndComponentCount(
                        decoded, StubOperandRole::kSource0, 1) == 1 &&
                    CountDescriptorsForRoleAndComponentCount(
                        decoded, StubOperandRole::kScale, 1) == 1 &&
                    CountDescriptorsForRoleAndComponentCount(
                        decoded, StubOperandRole::kPairedScale, 1) == 1 &&
                    CountDescriptorsForRoleAndComponentCount(
                        decoded, StubOperandRole::kDestination, 1) == 1 &&
                    CountSlotsOfKindAndValueClass(
                        decoded, StubOperandSlotKind::kSource0,
                        StubOperandValueClass::kVectorRegister) == 1 &&
                    CountSlotsOfKindAndValueClass(
                        decoded, StubOperandSlotKind::kScaleSource,
                        StubOperandValueClass::kScalarRegister) == 1 &&
                    CountSlotsOfKindAndValueClass(
                        decoded, StubOperandSlotKind::kPairedScaleSource,
                        StubOperandValueClass::kScalarRegister) == 1 &&
                    CountSlotsOfKindAndValueClass(
                        decoded, StubOperandSlotKind::kDestination,
                        StubOperandValueClass::kVectorRegister) == 1 &&
                    CountSlotsOfKindAndFragmentKind(
                        decoded, StubOperandSlotKind::kSource0,
                        StubFragmentKind::kVector) == 1 &&
                    CountSlotsOfKindAndFragmentKind(
                        decoded, StubOperandSlotKind::kScaleSource,
                        StubFragmentKind::kScalar) == 1 &&
                    CountSlotsOfKindAndFragmentKind(
                        decoded, StubOperandSlotKind::kPairedScaleSource,
                        StubFragmentKind::kScalar) == 1 &&
                    CountSlotsOfKindAndFragmentKind(
                        decoded, StubOperandSlotKind::kDestination,
                        StubFragmentKind::kVector) == 1 &&
                    CountSlotsOfKindAndComponentCount(
                        decoded, StubOperandSlotKind::kSource0, 1) == 1 &&
                    CountSlotsOfKindAndComponentCount(
                        decoded, StubOperandSlotKind::kScaleSource, 1) == 1 &&
                    CountSlotsOfKindAndComponentCount(
                        decoded, StubOperandSlotKind::kPairedScaleSource, 1) == 1 &&
                    CountSlotsOfKindAndComponentCount(
                        decoded, StubOperandSlotKind::kDestination, 1) == 1 &&
                    CountSlotsOfKindWithOutputFlag(
                        decoded, StubOperandSlotKind::kDestination, true) == 1 &&
                    CountSlotsOfKindWithOutputFlag(
                        decoded, StubOperandSlotKind::kSource0, false) == 1 &&
                    CountSlotsOfKindWithOutputFlag(
                        decoded, StubOperandSlotKind::kScaleSource, false) == 1 &&
                    CountSlotsOfKindWithOutputFlag(
                        decoded, StubOperandSlotKind::kPairedScaleSource,
                        false) == 1 &&
                    AllSlotsExplicit(decoded) &&
                    AllDescriptorsExplicit(decoded) &&
                    AllSlotWaveSizesAre(decoded, 0) &&
                    AllDescriptorWaveSizesAre(decoded, 0),
                "expected paired-scale helper to keep non-matrix wave-size semantics")) {
      return 1;
    }
    if (!Expect(
            CountDescriptorsForRoleAndSlotKind(
                decoded, StubOperandRole::kSource0,
                StubOperandSlotKind::kSource0) == 1 &&
                CountDescriptorsForRoleAndSlotKind(
                    decoded, StubOperandRole::kScale,
                    StubOperandSlotKind::kScaleSource) == 1 &&
                CountDescriptorsForRoleAndSlotKind(
                    decoded, StubOperandRole::kPairedScale,
                    StubOperandSlotKind::kPairedScaleSource) == 1 &&
                CountDescriptorsForRoleAndSlotKind(
                    decoded, StubOperandRole::kDestination,
                    StubOperandSlotKind::kDestination) == 1 &&
                CountSlotsOfKindAndLogicalOperandIndex(
                    decoded, StubOperandSlotKind::kDestination, 0) == 1 &&
                CountSlotsOfKindAndLogicalOperandIndex(
                    decoded, StubOperandSlotKind::kSource0, 1) == 1 &&
                CountSlotsOfKindAndLogicalOperandIndex(
                    decoded, StubOperandSlotKind::kScaleSource, 2) == 1 &&
                CountSlotsOfKindAndLogicalOperandIndex(
                    decoded, StubOperandSlotKind::kPairedScaleSource, 3) == 1,
            "expected paired-scale helper to keep exact role/slot and logical-index mapping")) {
      return 1;
    }
    if (!Expect(
            CountDescriptorsForRoleAndWaveSize(
                decoded, StubOperandRole::kSource0, 0) == 1 &&
                CountDescriptorsForRoleAndWaveSize(
                    decoded, StubOperandRole::kScale, 0) == 1 &&
                CountDescriptorsForRoleAndWaveSize(
                    decoded, StubOperandRole::kPairedScale, 0) == 1 &&
                CountDescriptorsForRoleAndWaveSize(
                    decoded, StubOperandRole::kDestination, 0) == 1 &&
                CountSlotsOfKindAndWaveSize(
                    decoded, StubOperandSlotKind::kDestination, 0) == 1 &&
                CountSlotsOfKindAndWaveSize(
                    decoded, StubOperandSlotKind::kSource0, 0) == 1 &&
                CountSlotsOfKindAndWaveSize(
                    decoded, StubOperandSlotKind::kScaleSource, 0) == 1 &&
                CountSlotsOfKindAndWaveSize(
                    decoded, StubOperandSlotKind::kPairedScaleSource, 0) == 1,
            "expected paired-scale helper to keep exact role/slot wave-size mapping")) {
      return 1;
    }
    const std::uint8_t paired_destination_element_bit_width =
        FindUniqueSlotElementBitWidth(decoded, StubOperandSlotKind::kDestination);
    const std::uint8_t paired_source0_element_bit_width =
        FindUniqueSlotElementBitWidth(decoded, StubOperandSlotKind::kSource0);
    if (!Expect(
            paired_destination_element_bit_width != 0xff &&
                paired_source0_element_bit_width != 0xff &&
                paired_destination_element_bit_width ==
                    paired_source0_element_bit_width &&
                CountDescriptorsForRoleAndElementBitWidth(
                    decoded, StubOperandRole::kSource0,
                    paired_source0_element_bit_width) == 1 &&
                CountDescriptorsForRoleAndElementBitWidth(
                    decoded, StubOperandRole::kScale, 32) == 1 &&
                CountDescriptorsForRoleAndElementBitWidth(
                    decoded, StubOperandRole::kPairedScale, 32) == 1 &&
                CountDescriptorsForRoleAndElementBitWidth(
                    decoded, StubOperandRole::kDestination,
                    paired_destination_element_bit_width) == 1 &&
                CountSlotsOfKindAndElementBitWidth(
                    decoded, StubOperandSlotKind::kSource0,
                    paired_source0_element_bit_width) == 1 &&
                CountSlotsOfKindAndElementBitWidth(
                    decoded, StubOperandSlotKind::kScaleSource, 32) == 1 &&
                CountSlotsOfKindAndElementBitWidth(
                    decoded, StubOperandSlotKind::kPairedScaleSource, 32) == 1 &&
                CountSlotsOfKindAndElementBitWidth(
                    decoded, StubOperandSlotKind::kDestination,
                    paired_destination_element_bit_width) == 1,
            "expected paired-scale helper to keep exact role/slot element-width mapping")) {
      return 1;
    }
    if (!Expect(
            CountDescriptorsForRoleAndPackedElements(
                decoded, StubOperandRole::kSource0, 1) == 1 &&
                CountDescriptorsForRoleAndPackedElements(
                    decoded, StubOperandRole::kScale, 1) == 1 &&
                CountDescriptorsForRoleAndPackedElements(
                    decoded, StubOperandRole::kPairedScale, 1) == 1 &&
                CountDescriptorsForRoleAndPackedElements(
                    decoded, StubOperandRole::kDestination, 1) == 1 &&
                CountSlotsOfKindAndPackedElements(
                    decoded, StubOperandSlotKind::kSource0, 1) == 1 &&
                CountSlotsOfKindAndPackedElements(
                    decoded, StubOperandSlotKind::kScaleSource, 1) == 1 &&
                CountSlotsOfKindAndPackedElements(
                    decoded, StubOperandSlotKind::kPairedScaleSource, 1) == 1 &&
                CountSlotsOfKindAndPackedElements(
                    decoded, StubOperandSlotKind::kDestination, 1) == 1,
            "expected paired-scale helper to keep exact role/slot packed-elements mapping")) {
      return 1;
    }
    if (!Expect(
            CountDescriptorsForRoleAndDimensions(
                decoded, StubOperandRole::kSource0, 1, 1, 1) == 1 &&
                CountDescriptorsForRoleAndDimensions(
                    decoded, StubOperandRole::kScale, 1, 1, 1) == 1 &&
                CountDescriptorsForRoleAndDimensions(
                    decoded, StubOperandRole::kPairedScale, 1, 1, 1) == 1 &&
                CountDescriptorsForRoleAndDimensions(
                    decoded, StubOperandRole::kDestination, 1, 1, 1) == 1 &&
                CountSlotsOfKindAndDimensions(
                    decoded, StubOperandSlotKind::kSource0, 1, 1, 1) == 1 &&
                CountSlotsOfKindAndDimensions(
                    decoded, StubOperandSlotKind::kScaleSource, 1, 1, 1) == 1 &&
                CountSlotsOfKindAndDimensions(
                    decoded, StubOperandSlotKind::kPairedScaleSource, 1, 1, 1) ==
                    1 &&
                CountSlotsOfKindAndDimensions(
                    decoded, StubOperandSlotKind::kDestination, 1, 1, 1) == 1,
            "expected paired-scale helper to keep exact role/slot dimension mapping")) {
      return 1;
    }
    if (!Expect(
            CountRoleBindings(decoded, StubOperandRole::kSource0) == 1 &&
                CountRoleBindings(decoded, StubOperandRole::kScale) == 1 &&
                CountRoleBindings(decoded, StubOperandRole::kPairedScale) == 1 &&
                CountRoleBindings(decoded, StubOperandRole::kDestination) == 1 &&
                CountRoleBindingsWithCount(decoded, StubOperandRole::kSource0,
                                           1) == 1 &&
                CountRoleBindingsWithCount(decoded, StubOperandRole::kScale,
                                           1) == 1 &&
                CountRoleBindingsWithCount(decoded,
                                           StubOperandRole::kPairedScale,
                                           1) == 1 &&
                CountRoleBindingsWithCount(
                    decoded, StubOperandRole::kDestination, 1) == 1 &&
                CountRoleBindingsWithOutputFlag(
                    decoded, StubOperandRole::kSource0, false) == 1 &&
                CountRoleBindingsWithOutputFlag(
                    decoded, StubOperandRole::kScale, false) == 1 &&
                CountRoleBindingsWithOutputFlag(
                    decoded, StubOperandRole::kPairedScale, false) == 1 &&
                CountRoleBindingsWithOutputFlag(
                    decoded, StubOperandRole::kDestination, true) == 1,
            "expected paired-scale helper to keep exact operand-role binding mapping")) {
      return 1;
    }
    if (!Expect(decoded.operand_roles.binding_count == 4 &&
                    AllRoleBindingsExplicit(decoded),
                "expected paired-scale helper to keep exact operand-role binding-count and explicitness")) {
      return 1;
    }
    if (!Expect(
            MatchesRoleBindingSequence(
                decoded,
                {{StubOperandRole::kSource0, 1, false, false},
                 {StubOperandRole::kScale, 1, false, false},
                 {StubOperandRole::kPairedScale, 1, false, false},
                 {StubOperandRole::kDestination, 1, true, false}}) &&
                MatchesSlotBindingSequence(
                    decoded,
                    {{StubOperandSlotKind::kDestination,
                      StubOperandValueClass::kVectorRegister,
                      0,
                      1,
                      true,
                      false},
                     {StubOperandSlotKind::kSource0,
                      StubOperandValueClass::kVectorRegister,
                      1,
                      1,
                      false,
                      false},
                     {StubOperandSlotKind::kScaleSource,
                      StubOperandValueClass::kScalarRegister,
                      2,
                      1,
                      false,
                      false},
                     {StubOperandSlotKind::kPairedScaleSource,
                      StubOperandValueClass::kScalarRegister,
                      3,
                      1,
                      false,
                      false}}) &&
                MatchesDescriptorSequence(
                    decoded,
                    {{StubOperandRole::kDestination,
                      StubOperandSlotKind::kDestination,
                      StubOperandValueClass::kVectorRegister,
                      StubOperandAccess::kWrite,
                      1,
                      false},
                     {StubOperandRole::kSource0,
                      StubOperandSlotKind::kSource0,
                      StubOperandValueClass::kVectorRegister,
                      StubOperandAccess::kRead,
                      1,
                      false},
                     {StubOperandRole::kScale,
                      StubOperandSlotKind::kScaleSource,
                      StubOperandValueClass::kScalarRegister,
                      StubOperandAccess::kRead,
                      1,
                      false},
                     {StubOperandRole::kPairedScale,
                      StubOperandSlotKind::kPairedScaleSource,
                      StubOperandValueClass::kScalarRegister,
                      StubOperandAccess::kRead,
                      1,
                      false}}),
            "expected paired-scale helper to keep exact operand-role, slot, and descriptor order")) {
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
  const StubDecoderRouteInfo synthetic_unsupported_info{
      "SYNTHETIC_GFX1250_UNSUPPORTED",
      StubDecoderRoute::kUnsupported,
      "kBogusRoute",
      99,
      DecodeSeedHint::kUnknown,
      "BOGUS_ENC",
      123,
      7,
      true,
      true,
  };
  const StubDecodedInstruction via_unsupported_info =
      DecodeStubInstruction(synthetic_unsupported_info);
  if (!Expect(MatchesUnsupportedInstructionDecode(
                  via_unsupported_info,
                  synthetic_unsupported_info.instruction_name),
              "expected unsupported route-info decode to normalize to exact unsupported shape")) {
    return 1;
  }
  const StubDecoderRouteInfo synthetic_invalid_route_info{
      "SYNTHETIC_GFX1250_INVALID_ROUTE",
      static_cast<StubDecoderRoute>(99),
      "kInvalidRoute",
      123,
      DecodeSeedHint::kVop3p,
      "INVALID_ENC",
      456,
      8,
      true,
      true,
  };
  const StubDecodedInstruction via_invalid_route_info =
      DecodeStubInstruction(synthetic_invalid_route_info);
  if (!Expect(MatchesUnsupportedInstructionDecode(
                  via_invalid_route_info,
                  synthetic_invalid_route_info.instruction_name),
              "expected invalid route-info decode to normalize to exact unsupported shape")) {
    return 1;
  }
  std::uint32_t unsupported_seed_count = 0;
  for (const DecoderSeedInfo& seed : GetDecoderSeedInfos()) {
    if (!IsUnsupportedSeededInstruction(seed)) {
      continue;
    }
    ++unsupported_seed_count;
    const StubDecodedInstruction decoded =
        DecodeStubInstruction(seed.instruction_name);
    if (!Expect(MatchesUnsupportedSeedDecode(decoded, seed) &&
                    MatchesUnknownHelperSurface(decoded) &&
                    MatchesUnsupportedInstructionDecode(decoded,
                                                        seed.instruction_name) &&
                    FindStubDecoderRouteInfo(seed.instruction_name) == nullptr,
                "expected unsupported seeded op to keep exact unsupported-route decode parity")) {
      return 1;
    }
    for (const StubDecoderRouteManifest& manifest : GetStubDecoderRouteManifests()) {
      const StubDecodedInstruction via_entrypoint =
          DecodeViaExplicitRouteEntrypoint(manifest.route,
                                          seed.instruction_name);
      if (!Expect(MatchesUnsupportedSeedDecode(via_entrypoint, seed) &&
                      MatchesUnknownHelperSurface(via_entrypoint),
                  "expected unsupported seeded op to keep exact route-keyed unsupported parity")) {
        return 1;
      }
    }
  }
  if (!Expect(unsupported_seed_count > 0,
              "expected at least one unsupported seeded op to validate")) {
    return 1;
  }
  bool found_clean_unsupported_seed = false;
  for (const DecoderSeedInfo& seed : GetDecoderSeedInfos()) {
    if (seed.instruction_name == "V_CVT_SCALEF32_PK8_FP8_F32" &&
        IsUnsupportedSeededInstruction(seed)) {
      found_clean_unsupported_seed = true;
      break;
    }
  }
  if (!Expect(found_clean_unsupported_seed,
              "expected clean unsupported seeded representative in gfx1250 seed catalog")) {
    return 1;
  }
  for (const DecoderSeedInfo& seed : GetDecoderSeedInfos()) {
    const StubDecodeStatus expected_status =
        IsUnsupportedSeededInstruction(seed) ? StubDecodeStatus::kUnsupportedRoute
                                             : StubDecodeStatus::kDecodedStub;
    if (!Expect(MatchesSelectorDecodeStatusParity(seed.instruction_name,
                                                  expected_status),
                "expected selector and decode surfaces to agree on routed vs unsupported seeded status")) {
      return 1;
    }
  }

  const StubDecodedInstruction unknown =
      DecodeStubInstruction("NO_SUCH_GFX1250_OPCODE");
  if (!Expect(MatchesUnknownDecode(unknown, "NO_SUCH_GFX1250_OPCODE") &&
                  MatchesUnknownHelperSurface(unknown),
              "expected exact unknown-instruction decode shape for missing opcode")) {
    return 1;
  }
  for (const StubDecoderRouteManifest& manifest : GetStubDecoderRouteManifests()) {
    const StubDecodedInstruction via_entrypoint =
        DecodeViaExplicitRouteEntrypoint(manifest.route,
                                        "NO_SUCH_GFX1250_OPCODE");
    if (!Expect(MatchesUnknownDecode(via_entrypoint, "NO_SUCH_GFX1250_OPCODE") &&
                    MatchesUnknownHelperSurface(via_entrypoint),
                "expected unknown opcode to keep exact route-keyed unknown parity")) {
      return 1;
    }
  }
  if (!Expect(MatchesSelectorDecodeStatusParity(
                  "NO_SUCH_GFX1250_OPCODE",
                  StubDecodeStatus::kUnknownInstruction) &&
                  MatchesSelectorDecodeStatusParity(
                      "v_pk_add_bf16",
                      StubDecodeStatus::kUnknownInstruction) &&
                  MatchesSelectorDecodeStatusParity(
                      " V_PK_ADD_BF16",
                      StubDecodeStatus::kUnknownInstruction) &&
                  MatchesSelectorDecodeStatusParity(
                      "V_PK_ADD_BF16 ",
                      StubDecodeStatus::kUnknownInstruction) &&
                  MatchesSelectorDecodeStatusParity(
                      "", StubDecodeStatus::kUnknownInstruction),
              "expected selector and decode surfaces to agree on unknown-name status")) {
    return 1;
  }
  const StubDecodedInstruction empty_instruction = DecodeStubInstruction("");
  if (!Expect(MatchesUnknownDecode(empty_instruction, "") &&
                  MatchesUnknownHelperSurface(empty_instruction),
              "expected empty instruction name to keep exact unknown decode parity")) {
    return 1;
  }
  for (const StubDecoderRouteManifest& manifest : GetStubDecoderRouteManifests()) {
    const StubDecodedInstruction via_entrypoint =
        DecodeViaExplicitRouteEntrypoint(manifest.route, "");
    if (!Expect(MatchesUnknownDecode(via_entrypoint, "") &&
                    MatchesUnknownHelperSurface(via_entrypoint),
                "expected empty instruction name to keep exact route-keyed unknown parity")) {
      return 1;
    }
  }
  const StubDecodedInstruction lowercase_instruction =
      DecodeStubInstruction("v_pk_add_bf16");
  if (!Expect(MatchesUnknownDecode(lowercase_instruction, "v_pk_add_bf16") &&
                  MatchesUnknownHelperSurface(lowercase_instruction),
              "expected lowercase known opcode to keep exact unknown decode parity")) {
    return 1;
  }
  for (const StubDecoderRouteManifest& manifest : GetStubDecoderRouteManifests()) {
    const StubDecodedInstruction via_entrypoint =
        DecodeViaExplicitRouteEntrypoint(manifest.route, "v_pk_add_bf16");
    if (!Expect(MatchesUnknownDecode(via_entrypoint, "v_pk_add_bf16") &&
                    MatchesUnknownHelperSurface(via_entrypoint),
                "expected lowercase known opcode to keep exact route-keyed unknown parity")) {
      return 1;
    }
  }
  for (std::string_view padded_instruction :
       {" V_PK_ADD_BF16", "V_PK_ADD_BF16 "}) {
    const StubDecodedInstruction padded_decode =
        DecodeStubInstruction(padded_instruction);
    if (!Expect(MatchesUnknownDecode(padded_decode, padded_instruction) &&
                    MatchesUnknownHelperSurface(padded_decode),
                "expected whitespace-padded known opcode to keep exact unknown decode parity")) {
      return 1;
    }
    for (const StubDecoderRouteManifest& manifest :
         GetStubDecoderRouteManifests()) {
      const StubDecodedInstruction via_entrypoint =
          DecodeViaExplicitRouteEntrypoint(manifest.route, padded_instruction);
      if (!Expect(MatchesUnknownDecode(via_entrypoint, padded_instruction) &&
                      MatchesUnknownHelperSurface(via_entrypoint),
                  "expected whitespace-padded known opcode to keep exact route-keyed unknown parity")) {
        return 1;
      }
    }
  }
  if (!Expect(GetStubOpcodeShapeName(static_cast<StubOpcodeShape>(99)) ==
                      "kUnknown" &&
                  GetStubExecutionDomainName(
                      static_cast<StubExecutionDomain>(99)) == "kUnknown" &&
                  GetStubOperandLayoutName(
                      static_cast<StubOperandLayoutKind>(99)) == "kUnknown" &&
                  GetStubOperandRoleName(static_cast<StubOperandRole>(99)) ==
                      "kUnknown" &&
                  GetStubOperandSlotKindName(
                      static_cast<StubOperandSlotKind>(99)) == "kUnknown" &&
                  GetStubOperandValueClassName(
                      static_cast<StubOperandValueClass>(99)) == "kUnknown",
              "expected helper-name fallbacks for invalid gfx1250 stub enums")) {
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
  if (!Expect(FindStubDecoderEntrypointManifest(StubDecoderRoute::kUnsupported) ==
                  nullptr,
              "expected no entrypoint manifest for unsupported route")) {
    return 1;
  }
  if (!Expect(FindStubDecoderEntrypointManifest(
                  static_cast<StubDecoderRoute>(99)) == nullptr,
              "expected no entrypoint manifest for invalid route")) {
    return 1;
  }

  const auto entrypoint_manifests = GetStubDecoderEntrypointManifests();
  if (!Expect(entrypoint_manifests.size() == 4 &&
                  entrypoint_manifests[0].route == StubDecoderRoute::kVop3p &&
                  entrypoint_manifests[0].route_priority == 1 &&
                  entrypoint_manifests[0].entrypoint_name == "DecodeVop3pStub" &&
                  entrypoint_manifests[1].route ==
                      StubDecoderRoute::kMimgTensor &&
                  entrypoint_manifests[1].route_priority == 2 &&
                  entrypoint_manifests[1].entrypoint_name ==
                      "DecodeMimgTensorStub" &&
                  entrypoint_manifests[2].route == StubDecoderRoute::kVop1 &&
                  entrypoint_manifests[2].route_priority == 3 &&
                  entrypoint_manifests[2].entrypoint_name == "DecodeVop1Stub" &&
                  entrypoint_manifests[3].route ==
                      StubDecoderRoute::kVop3Sdst &&
                  entrypoint_manifests[3].route_priority == 4 &&
                  entrypoint_manifests[3].entrypoint_name ==
                      "DecodeVop3SdstStub",
              "expected entrypoint manifests to stay in exact route-priority order")) {
    return 1;
  }

  std::size_t total_manifest_instructions = 0;
  for (const StubDecoderEntrypointManifest& manifest : entrypoint_manifests) {
    total_manifest_instructions += manifest.instruction_count;
  }
  if (!Expect(total_manifest_instructions ==
                  GetStubDecoderRouteInfos().size(),
              "expected entrypoint manifests to cover all routed seeds")) {
    return 1;
  }
  if (!Expect(EntrypointManifestLookupMatchesSequenceEntries(),
              "expected entrypoint manifest lookup to return sequence-stable entries")) {
    return 1;
  }
  if (!Expect(EntrypointAndRouteManifestSequencesMatchExactly(),
              "expected entrypoint and route manifest sequences to match exactly by route metadata and count")) {
    return 1;
  }
  if (!Expect(EntrypointManifestCountsMatchRoutedSurfaces(),
              "expected entrypoint manifest counts to match routed instruction lists and route infos per route")) {
    return 1;
  }

  if (!Expect(GetStubDecoderRouteManifests().size() == 4,
              "expected four route manifests")) {
    return 1;
  }
  const auto route_manifests = GetStubDecoderRouteManifests();
  if (!Expect(route_manifests.size() == 4 &&
                  route_manifests[0].route == StubDecoderRoute::kVop3p &&
                  route_manifests[0].route_name == "kVop3p" &&
                  route_manifests[0].route_priority == 1 &&
                  route_manifests[1].route == StubDecoderRoute::kMimgTensor &&
                  route_manifests[1].route_name == "kMimgTensor" &&
                  route_manifests[1].route_priority == 2 &&
                  route_manifests[2].route == StubDecoderRoute::kVop1 &&
                  route_manifests[2].route_name == "kVop1" &&
                  route_manifests[2].route_priority == 3 &&
                  route_manifests[3].route == StubDecoderRoute::kVop3Sdst &&
                  route_manifests[3].route_name == "kVop3Sdst" &&
                  route_manifests[3].route_priority == 4,
              "expected route manifests to stay in exact route-priority order")) {
    return 1;
  }
  if (!Expect(RouteManifestLookupMatchesSequenceEntries(),
              "expected route manifest lookup to return sequence-stable entries")) {
    return 1;
  }
  if (!Expect(GlobalRouteInfoSequenceMatchesRouteInstructionLists(),
              "expected global route-info sequence to match manifest-ordered routed instruction lists")) {
    return 1;
  }
  if (!Expect(RoutedInstructionNamesFormUniqueBijection(),
              "expected routed instruction names to form a unique bijection between route infos and routed instruction lists")) {
    return 1;
  }
  for (const StubDecoderRouteManifest& manifest : route_manifests) {
    if (!Expect(
            manifest.instruction_count ==
                    GetStubDecoderRouteInstructions(manifest.route).size() &&
                manifest.instruction_count ==
                    CountRouteInfosForRoute(manifest.route) &&
                manifest.xml_backed_count ==
                    CountRouteInfosForRouteWithXmlFlag(manifest.route, true) &&
                manifest.llvm_only_count ==
                    CountRouteInfosForRouteWithXmlFlag(manifest.route, false) &&
                manifest.target_specific_count ==
                    CountRouteInfosForRouteWithTargetSpecificFlag(
                        manifest.route, true) &&
                manifest.xml_backed_count + manifest.llvm_only_count ==
                    manifest.instruction_count,
            "expected route manifest counts to match routed instruction and provenance totals")) {
      return 1;
    }
    if (!Expect(RouteInstructionListMatchesRouteInfoSequence(manifest.route),
                "expected routed instruction list order to match route-info sequence")) {
      return 1;
    }
  }
  if (!Expect(RouteInfoLookupMatchesSequenceEntries(),
              "expected route-info lookup to return sequence-stable entries")) {
    return 1;
  }

  const StubDecoderRouteManifest* vop3p_route_manifest =
      FindStubDecoderRouteManifest(StubDecoderRoute::kVop3p);
  if (!Expect(vop3p_route_manifest != nullptr,
              "expected VOP3P route manifest")) {
    return 1;
  }
  if (!Expect(vop3p_route_manifest->route_name == "kVop3p" &&
                  vop3p_route_manifest->route_priority == 1,
              "expected VOP3P route manifest metadata")) {
    return 1;
  }
  const StubDecoderRouteManifest* tensor_route_manifest =
      FindStubDecoderRouteManifest(StubDecoderRoute::kMimgTensor);
  if (!Expect(tensor_route_manifest != nullptr,
              "expected tensor route manifest")) {
    return 1;
  }
  if (!Expect(tensor_route_manifest->route_name == "kMimgTensor" &&
                  tensor_route_manifest->route_priority == 2,
              "expected tensor route manifest metadata")) {
    return 1;
  }
  const StubDecoderRouteManifest* vop1_route_manifest =
      FindStubDecoderRouteManifest(StubDecoderRoute::kVop1);
  if (!Expect(vop1_route_manifest != nullptr,
              "expected VOP1 route manifest")) {
    return 1;
  }
  if (!Expect(vop1_route_manifest->route_name == "kVop1" &&
                  vop1_route_manifest->route_priority == 3,
              "expected VOP1 route manifest metadata")) {
    return 1;
  }
  const StubDecoderRouteManifest* sdst_route_manifest =
      FindStubDecoderRouteManifest(StubDecoderRoute::kVop3Sdst);
  if (!Expect(sdst_route_manifest != nullptr,
              "expected VOP3 SDST route manifest")) {
    return 1;
  }
  if (!Expect(sdst_route_manifest->route_name == "kVop3Sdst" &&
                  sdst_route_manifest->route_priority == 4,
              "expected VOP3 SDST route manifest metadata")) {
    return 1;
  }
  if (!Expect(FindStubDecoderRouteManifest(StubDecoderRoute::kUnsupported) ==
                  nullptr,
              "expected no route manifest for unsupported route")) {
    return 1;
  }
  if (!Expect(FindStubDecoderRouteManifest(static_cast<StubDecoderRoute>(99)) ==
                  nullptr,
              "expected no route manifest for invalid route")) {
    return 1;
  }

  for (const StubDecoderRouteInfo& route_info : GetStubDecoderRouteInfos()) {
    const StubDecodedInstruction via_name =
        DecodeStubInstruction(route_info.instruction_name);
    const StubDecodedInstruction via_info = DecodeStubInstruction(route_info);
    if (!Expect(MatchesDecodedInstruction(via_name, via_info),
                "expected route-info-based decode to match name-based decode across routed seeds")) {
      return 1;
    }
    const StubDecodedInstruction via_entrypoint =
        DecodeViaRouteEntrypoint(route_info);
    if (!Expect(MatchesDecodedInstruction(via_name, via_entrypoint),
                "expected route-keyed entrypoint decode to match name-based decode across routed seeds")) {
      return 1;
    }
    const StubDecoderEntrypointManifest* entrypoint_manifest =
        FindStubDecoderEntrypointManifest(via_name.route);
    const StubDecoderRouteManifest* route_manifest =
        FindStubDecoderRouteManifest(via_name.route);
    if (!Expect(entrypoint_manifest != nullptr && route_manifest != nullptr &&
                    via_name.route == entrypoint_manifest->route &&
                    via_name.route == route_manifest->route &&
                    via_name.route_name == route_manifest->route_name &&
                    via_name.route_priority == route_manifest->route_priority &&
                    via_name.entrypoint_name ==
                        entrypoint_manifest->entrypoint_name &&
                    via_name.route_priority ==
                        entrypoint_manifest->route_priority &&
                    route_manifest->instruction_count ==
                        entrypoint_manifest->instruction_count &&
                    IsInstructionListedForRoute(via_name.route,
                                                via_name.instruction_name),
                "expected decoded routed seed to match route and entrypoint manifest surfaces")) {
      return 1;
    }
    if (!Expect(GetStubOpcodeShapeName(via_name.opcode_shape) ==
                        ExpectedOpcodeShapeName(via_name.instruction_name) &&
                    GetStubExecutionDomainName(via_name.execution_domain) ==
                        ExpectedExecutionDomainName(via_name.instruction_name) &&
                    GetStubOperandLayoutName(via_name.operand_layout.layout_kind) ==
                        ExpectedOperandLayoutName(via_name.instruction_name) &&
                    AllRoleHelperNamesKnown(via_name) &&
                    AllSlotKindHelperNamesKnown(via_name) &&
                    AllValueClassHelperNamesKnown(via_name),
                "expected routed seed to keep exact helper-name parity and coverage")) {
      return 1;
    }
  }
  for (const StubDecoderRouteInfo& route_info : GetStubDecoderRouteInfos()) {
    StubDecoderRouteInfo synthetic = route_info;
    synthetic.route_name = "kSyntheticRoute";
    synthetic.route_priority = route_info.route_priority + 100u;
    synthetic.rdna4_encoding_name = "SYNTHETIC_ENC";
    synthetic.rdna4_opcode = route_info.rdna4_opcode + 1000u;
    synthetic.rdna4_operand_count = route_info.rdna4_operand_count + 10u;
    synthetic.appears_in_rdna4_xml = !route_info.appears_in_rdna4_xml;
    synthetic.is_target_specific = !route_info.is_target_specific;

    const StubDecodedInstruction via_name =
        DecodeStubInstruction(route_info.instruction_name);
    const StubDecodedInstruction via_synthetic =
        DecodeStubInstruction(synthetic);
    if (!Expect(MatchesRouteInfoPayload(via_synthetic, synthetic) &&
                    MatchesDecodedInstructionStructure(via_synthetic,
                                                      via_name),
                "expected route-info decode to preserve caller metadata while keeping structural parity")) {
      return 1;
    }
    StubDecoderRouteInfo synthetic_invalid_hint = synthetic;
    synthetic_invalid_hint.route_name = "kSyntheticRouteInvalidHint";
    synthetic_invalid_hint.route_priority = route_info.route_priority + 150u;
    synthetic_invalid_hint.decode_hint = static_cast<DecodeSeedHint>(99);
    const StubDecodedInstruction via_invalid_hint =
        DecodeStubInstruction(synthetic_invalid_hint);
    if (!Expect(
            MatchesRouteInfoPayload(via_invalid_hint, synthetic_invalid_hint) &&
                MatchesDecodedInstructionStructure(via_invalid_hint, via_name),
            "expected routed route-info decode to ignore invalid caller decode-hint while preserving metadata")) {
      return 1;
    }
    StubDecoderRouteInfo synthetic_valid_wrong_hint = synthetic;
    synthetic_valid_wrong_hint.route_name = "kSyntheticRouteValidWrongHint";
    synthetic_valid_wrong_hint.route_priority =
        route_info.route_priority + 160u;
    synthetic_valid_wrong_hint.decode_hint =
        AlternateDecodeHintForRoute(route_info.route);
    const StubDecodedInstruction via_valid_wrong_hint =
        DecodeStubInstruction(synthetic_valid_wrong_hint);
    if (!Expect(
            MatchesRouteInfoPayload(via_valid_wrong_hint,
                                    synthetic_valid_wrong_hint) &&
                MatchesDecodedInstructionStructure(via_valid_wrong_hint,
                                                  via_name),
            "expected routed route-info decode to ignore valid mismatching caller decode-hint while preserving metadata")) {
      return 1;
    }
  }
  for (const StubDecoderEntrypointManifest& manifest :
       GetStubDecoderEntrypointManifests()) {
    const StubDecoderRouteInfo synthetic_unknown{
        SyntheticUnknownInstructionForRoute(manifest.route),
        manifest.route,
        "kSyntheticRouteInfoUnknown",
        manifest.route_priority + 100u,
        DecodeSeedHint::kUnknown,
        "SYNTHETIC_UNKNOWN_ENC",
        1000u + manifest.route_priority,
        10u + manifest.route_priority,
        true,
        true,
    };
    const StubDecodedInstruction decoded =
        DecodeStubInstruction(synthetic_unknown);
    if (!Expect(decoded.status == StubDecodeStatus::kDecodedStub &&
                    MatchesRouteInfoPayload(decoded, synthetic_unknown) &&
                    decoded.entrypoint_name == manifest.entrypoint_name &&
                    MatchesUnknownHelperSurface(decoded) &&
                    MatchesTopLevelFlags(decoded, false, false, false, false) &&
                    MatchesLayout(decoded, ExpectedLayout{}) &&
                    decoded.operand_roles.binding_count == 0 &&
                    decoded.operand_slots.binding_count == 0 &&
                    decoded.operand_descriptors.descriptor_count == 0,
                "expected synthetic routed route-info with unknown instruction name to preserve caller metadata while keeping empty unknown structure")) {
      return 1;
    }
    const StubDecoderRouteInfo synthetic_empty{
        "",
        manifest.route,
        "kSyntheticRouteInfoEmpty",
        manifest.route_priority + 125u,
        DecodeSeedHint::kUnknown,
        "SYNTHETIC_EMPTY_ENC",
        1500u + manifest.route_priority,
        15u + manifest.route_priority,
        false,
        false,
    };
    const StubDecodedInstruction empty_decoded =
        DecodeStubInstruction(synthetic_empty);
    if (!Expect(empty_decoded.status == StubDecodeStatus::kDecodedStub &&
                    MatchesRouteInfoPayload(empty_decoded, synthetic_empty) &&
                    empty_decoded.entrypoint_name == manifest.entrypoint_name &&
                    MatchesUnknownHelperSurface(empty_decoded) &&
                    MatchesTopLevelFlags(empty_decoded,
                                         false,
                                         false,
                                         false,
                                         false) &&
                    MatchesLayout(empty_decoded, ExpectedLayout{}) &&
                    empty_decoded.operand_roles.binding_count == 0 &&
                    empty_decoded.operand_slots.binding_count == 0 &&
                    empty_decoded.operand_descriptors.descriptor_count == 0,
                "expected synthetic routed route-info with empty instruction name to preserve caller metadata while keeping empty unknown structure")) {
      return 1;
    }
    StubDecoderRouteInfo synthetic_empty_valid_wrong_hint = synthetic_empty;
    synthetic_empty_valid_wrong_hint.route_name =
        "kSyntheticRouteInfoEmptyValidWrongHint";
    synthetic_empty_valid_wrong_hint.route_priority =
        manifest.route_priority + 130u;
    synthetic_empty_valid_wrong_hint.decode_hint =
        AlternateDecodeHintForRoute(manifest.route);
    const StubDecodedInstruction empty_valid_wrong_hint_decoded =
        DecodeStubInstruction(synthetic_empty_valid_wrong_hint);
    if (!Expect(
            empty_valid_wrong_hint_decoded.status ==
                    StubDecodeStatus::kDecodedStub &&
                MatchesRouteInfoPayload(empty_valid_wrong_hint_decoded,
                                        synthetic_empty_valid_wrong_hint) &&
                empty_valid_wrong_hint_decoded.entrypoint_name ==
                    manifest.entrypoint_name &&
                MatchesUnknownHelperSurface(
                    empty_valid_wrong_hint_decoded) &&
                MatchesTopLevelFlags(empty_valid_wrong_hint_decoded,
                                     false,
                                     false,
                                     false,
                                     false) &&
                MatchesLayout(empty_valid_wrong_hint_decoded,
                              ExpectedLayout{}) &&
                empty_valid_wrong_hint_decoded.operand_roles.binding_count ==
                    0 &&
                empty_valid_wrong_hint_decoded.operand_slots.binding_count ==
                    0 &&
                empty_valid_wrong_hint_decoded.operand_descriptors
                        .descriptor_count == 0,
            "expected synthetic routed route-info with empty instruction name to ignore valid mismatching caller decode-hint while keeping empty unknown structure")) {
      return 1;
    }
    StubDecoderRouteInfo synthetic_empty_invalid_hint = synthetic_empty;
    synthetic_empty_invalid_hint.route_name =
        "kSyntheticRouteInfoEmptyInvalidHint";
    synthetic_empty_invalid_hint.route_priority =
        manifest.route_priority + 135u;
    synthetic_empty_invalid_hint.decode_hint =
        static_cast<DecodeSeedHint>(99);
    const StubDecodedInstruction empty_invalid_hint_decoded =
        DecodeStubInstruction(synthetic_empty_invalid_hint);
    if (!Expect(
            empty_invalid_hint_decoded.status == StubDecodeStatus::kDecodedStub &&
                MatchesRouteInfoPayload(empty_invalid_hint_decoded,
                                        synthetic_empty_invalid_hint) &&
                empty_invalid_hint_decoded.entrypoint_name ==
                    manifest.entrypoint_name &&
                MatchesUnknownHelperSurface(empty_invalid_hint_decoded) &&
                MatchesTopLevelFlags(empty_invalid_hint_decoded,
                                     false,
                                     false,
                                     false,
                                     false) &&
                MatchesLayout(empty_invalid_hint_decoded, ExpectedLayout{}) &&
                empty_invalid_hint_decoded.operand_roles.binding_count == 0 &&
                empty_invalid_hint_decoded.operand_slots.binding_count == 0 &&
                empty_invalid_hint_decoded.operand_descriptors
                        .descriptor_count == 0,
            "expected synthetic routed route-info with empty instruction name to ignore invalid caller decode-hint while keeping empty unknown structure")) {
      return 1;
    }
    const StubDecoderRouteInfo synthetic_unsupported_seeded{
        "V_CVT_SCALEF32_PK8_FP8_F32",
        manifest.route,
        "kSyntheticRouteInfoUnsupportedSeeded",
        manifest.route_priority + 175u,
        DecodeSeedHint::kUnknown,
        "SYNTHETIC_UNSUPPORTED_SEEDED_ENC",
        1750u + manifest.route_priority,
        17u + manifest.route_priority,
        false,
        false,
    };
    const StubDecodedInstruction unsupported_seeded_decoded =
        DecodeStubInstruction(synthetic_unsupported_seeded);
    if (!Expect(
            unsupported_seeded_decoded.status == StubDecodeStatus::kDecodedStub &&
                MatchesRouteInfoPayload(unsupported_seeded_decoded,
                                        synthetic_unsupported_seeded) &&
                unsupported_seeded_decoded.entrypoint_name ==
                    manifest.entrypoint_name &&
                MatchesUnknownHelperSurface(unsupported_seeded_decoded) &&
                MatchesTopLevelFlags(unsupported_seeded_decoded,
                                     false,
                                     false,
                                     false,
                                     false) &&
                MatchesLayout(unsupported_seeded_decoded, ExpectedLayout{}) &&
                unsupported_seeded_decoded.operand_roles.binding_count == 0 &&
                unsupported_seeded_decoded.operand_slots.binding_count == 0 &&
                unsupported_seeded_decoded.operand_descriptors
                        .descriptor_count == 0,
            "expected synthetic routed route-info with clean unsupported seeded instruction to preserve caller metadata while keeping empty unknown structure")) {
      return 1;
    }
    StubDecoderRouteInfo synthetic_unsupported_seeded_valid_wrong_hint =
        synthetic_unsupported_seeded;
    synthetic_unsupported_seeded_valid_wrong_hint.route_name =
        "kSyntheticRouteInfoUnsupportedSeededValidWrongHint";
    synthetic_unsupported_seeded_valid_wrong_hint.route_priority =
        manifest.route_priority + 180u;
    synthetic_unsupported_seeded_valid_wrong_hint.decode_hint =
        AlternateDecodeHintForRoute(manifest.route);
    const StubDecodedInstruction unsupported_seeded_valid_wrong_hint_decoded =
        DecodeStubInstruction(synthetic_unsupported_seeded_valid_wrong_hint);
    if (!Expect(
            unsupported_seeded_valid_wrong_hint_decoded.status ==
                    StubDecodeStatus::kDecodedStub &&
                MatchesRouteInfoPayload(
                    unsupported_seeded_valid_wrong_hint_decoded,
                    synthetic_unsupported_seeded_valid_wrong_hint) &&
                unsupported_seeded_valid_wrong_hint_decoded.entrypoint_name ==
                    manifest.entrypoint_name &&
                MatchesUnknownHelperSurface(
                    unsupported_seeded_valid_wrong_hint_decoded) &&
                MatchesTopLevelFlags(
                    unsupported_seeded_valid_wrong_hint_decoded,
                    false,
                    false,
                    false,
                    false) &&
                MatchesLayout(unsupported_seeded_valid_wrong_hint_decoded,
                              ExpectedLayout{}) &&
                unsupported_seeded_valid_wrong_hint_decoded.operand_roles
                        .binding_count == 0 &&
                unsupported_seeded_valid_wrong_hint_decoded.operand_slots
                        .binding_count == 0 &&
                unsupported_seeded_valid_wrong_hint_decoded
                        .operand_descriptors.descriptor_count == 0,
            "expected synthetic routed route-info with clean unsupported seeded instruction to ignore valid mismatching caller decode-hint while keeping empty unknown structure")) {
      return 1;
    }
    StubDecoderRouteInfo synthetic_unsupported_seeded_invalid_hint =
        synthetic_unsupported_seeded;
    synthetic_unsupported_seeded_invalid_hint.route_name =
        "kSyntheticRouteInfoUnsupportedSeededInvalidHint";
    synthetic_unsupported_seeded_invalid_hint.route_priority =
        manifest.route_priority + 185u;
    synthetic_unsupported_seeded_invalid_hint.decode_hint =
        static_cast<DecodeSeedHint>(99);
    const StubDecodedInstruction unsupported_seeded_invalid_hint_decoded =
        DecodeStubInstruction(synthetic_unsupported_seeded_invalid_hint);
    if (!Expect(
            unsupported_seeded_invalid_hint_decoded.status ==
                    StubDecodeStatus::kDecodedStub &&
                MatchesRouteInfoPayload(
                    unsupported_seeded_invalid_hint_decoded,
                    synthetic_unsupported_seeded_invalid_hint) &&
                unsupported_seeded_invalid_hint_decoded.entrypoint_name ==
                    manifest.entrypoint_name &&
                MatchesUnknownHelperSurface(
                    unsupported_seeded_invalid_hint_decoded) &&
                MatchesTopLevelFlags(unsupported_seeded_invalid_hint_decoded,
                                     false,
                                     false,
                                     false,
                                     false) &&
                MatchesLayout(unsupported_seeded_invalid_hint_decoded,
                              ExpectedLayout{}) &&
                unsupported_seeded_invalid_hint_decoded.operand_roles
                        .binding_count == 0 &&
                unsupported_seeded_invalid_hint_decoded.operand_slots
                        .binding_count == 0 &&
                unsupported_seeded_invalid_hint_decoded.operand_descriptors
                        .descriptor_count == 0,
            "expected synthetic routed route-info with clean unsupported seeded instruction to ignore invalid caller decode-hint while keeping empty unknown structure")) {
      return 1;
    }
    StubDecoderRouteInfo synthetic_unknown_invalid_hint = synthetic_unknown;
    synthetic_unknown_invalid_hint.route_name =
        "kSyntheticRouteInfoUnknownInvalidHint";
    synthetic_unknown_invalid_hint.route_priority =
        manifest.route_priority + 150u;
    synthetic_unknown_invalid_hint.decode_hint =
        static_cast<DecodeSeedHint>(99);
    const StubDecodedInstruction unknown_invalid_hint_decoded =
        DecodeStubInstruction(synthetic_unknown_invalid_hint);
    if (!Expect(
            unknown_invalid_hint_decoded.status == StubDecodeStatus::kDecodedStub &&
                MatchesRouteInfoPayload(unknown_invalid_hint_decoded,
                                        synthetic_unknown_invalid_hint) &&
                unknown_invalid_hint_decoded.entrypoint_name ==
                    manifest.entrypoint_name &&
                MatchesUnknownHelperSurface(unknown_invalid_hint_decoded) &&
                MatchesTopLevelFlags(unknown_invalid_hint_decoded,
                                     false,
                                     false,
                                     false,
                                     false) &&
                MatchesLayout(unknown_invalid_hint_decoded, ExpectedLayout{}) &&
                unknown_invalid_hint_decoded.operand_roles.binding_count == 0 &&
                unknown_invalid_hint_decoded.operand_slots.binding_count == 0 &&
                unknown_invalid_hint_decoded.operand_descriptors
                        .descriptor_count == 0,
            "expected synthetic routed route-info with invalid caller decode-hint to preserve metadata while keeping empty unknown structure")) {
      return 1;
    }
    StubDecoderRouteInfo synthetic_unknown_valid_wrong_hint =
        synthetic_unknown;
    synthetic_unknown_valid_wrong_hint.route_name =
        "kSyntheticRouteInfoUnknownValidWrongHint";
    synthetic_unknown_valid_wrong_hint.route_priority =
        manifest.route_priority + 145u;
    synthetic_unknown_valid_wrong_hint.decode_hint =
        AlternateDecodeHintForRoute(manifest.route);
    const StubDecodedInstruction unknown_valid_wrong_hint_decoded =
        DecodeStubInstruction(synthetic_unknown_valid_wrong_hint);
    if (!Expect(
            unknown_valid_wrong_hint_decoded.status ==
                    StubDecodeStatus::kDecodedStub &&
                MatchesRouteInfoPayload(unknown_valid_wrong_hint_decoded,
                                        synthetic_unknown_valid_wrong_hint) &&
                unknown_valid_wrong_hint_decoded.entrypoint_name ==
                    manifest.entrypoint_name &&
                MatchesUnknownHelperSurface(
                    unknown_valid_wrong_hint_decoded) &&
                MatchesTopLevelFlags(unknown_valid_wrong_hint_decoded,
                                     false,
                                     false,
                                     false,
                                     false) &&
                MatchesLayout(unknown_valid_wrong_hint_decoded,
                              ExpectedLayout{}) &&
                unknown_valid_wrong_hint_decoded.operand_roles.binding_count ==
                    0 &&
                unknown_valid_wrong_hint_decoded.operand_slots.binding_count ==
                    0 &&
                unknown_valid_wrong_hint_decoded.operand_descriptors
                        .descriptor_count == 0,
            "expected synthetic routed route-info with valid mismatching caller decode-hint to preserve metadata while keeping empty unknown structure")) {
      return 1;
    }
    for (std::string_view near_miss_instruction :
         {"v_pk_add_bf16", " V_PK_ADD_BF16"}) {
      const StubDecoderRouteInfo synthetic_near_miss{
          near_miss_instruction,
          manifest.route,
          "kSyntheticRouteInfoNearMiss",
          manifest.route_priority + 200u,
          DecodeSeedHint::kUnknown,
          "SYNTHETIC_NEAR_MISS_ENC",
          2000u + manifest.route_priority,
          20u + manifest.route_priority,
          false,
          false,
      };
      const StubDecodedInstruction near_miss_decoded =
          DecodeStubInstruction(synthetic_near_miss);
      if (!Expect(near_miss_decoded.status == StubDecodeStatus::kDecodedStub &&
                      MatchesRouteInfoPayload(near_miss_decoded,
                                              synthetic_near_miss) &&
                      near_miss_decoded.entrypoint_name ==
                          manifest.entrypoint_name &&
                      MatchesUnknownHelperSurface(near_miss_decoded) &&
                      MatchesTopLevelFlags(near_miss_decoded,
                                           false,
                                           false,
                                           false,
                                           false) &&
                      MatchesLayout(near_miss_decoded, ExpectedLayout{}) &&
                      near_miss_decoded.operand_roles.binding_count == 0 &&
                      near_miss_decoded.operand_slots.binding_count == 0 &&
                      near_miss_decoded.operand_descriptors.descriptor_count ==
                          0,
                  "expected synthetic routed route-info with near-miss known opcode to preserve caller metadata while keeping empty unknown structure")) {
        return 1;
      }
      StubDecoderRouteInfo synthetic_near_miss_valid_wrong_hint =
          synthetic_near_miss;
      synthetic_near_miss_valid_wrong_hint.route_name =
          "kSyntheticRouteInfoNearMissValidWrongHint";
      synthetic_near_miss_valid_wrong_hint.route_priority =
          manifest.route_priority + 215u;
      synthetic_near_miss_valid_wrong_hint.decode_hint =
          AlternateDecodeHintForRoute(manifest.route);
      const StubDecodedInstruction near_miss_valid_wrong_hint_decoded =
          DecodeStubInstruction(synthetic_near_miss_valid_wrong_hint);
      if (!Expect(
              near_miss_valid_wrong_hint_decoded.status ==
                      StubDecodeStatus::kDecodedStub &&
                  MatchesRouteInfoPayload(near_miss_valid_wrong_hint_decoded,
                                          synthetic_near_miss_valid_wrong_hint) &&
                  near_miss_valid_wrong_hint_decoded.entrypoint_name ==
                      manifest.entrypoint_name &&
                  MatchesUnknownHelperSurface(
                      near_miss_valid_wrong_hint_decoded) &&
                  MatchesTopLevelFlags(near_miss_valid_wrong_hint_decoded,
                                       false,
                                       false,
                                       false,
                                       false) &&
                  MatchesLayout(near_miss_valid_wrong_hint_decoded,
                                ExpectedLayout{}) &&
                  near_miss_valid_wrong_hint_decoded.operand_roles
                          .binding_count == 0 &&
                  near_miss_valid_wrong_hint_decoded.operand_slots
                          .binding_count == 0 &&
                  near_miss_valid_wrong_hint_decoded.operand_descriptors
                          .descriptor_count == 0,
              "expected synthetic routed route-info with near-miss known opcode to ignore valid mismatching caller decode-hint while keeping empty unknown structure")) {
        return 1;
      }
      StubDecoderRouteInfo synthetic_near_miss_invalid_hint =
          synthetic_near_miss;
      synthetic_near_miss_invalid_hint.route_name =
          "kSyntheticRouteInfoNearMissInvalidHint";
      synthetic_near_miss_invalid_hint.route_priority =
          manifest.route_priority + 225u;
      synthetic_near_miss_invalid_hint.decode_hint =
          static_cast<DecodeSeedHint>(99);
      const StubDecodedInstruction near_miss_invalid_hint_decoded =
          DecodeStubInstruction(synthetic_near_miss_invalid_hint);
      if (!Expect(
              near_miss_invalid_hint_decoded.status ==
                      StubDecodeStatus::kDecodedStub &&
                  MatchesRouteInfoPayload(near_miss_invalid_hint_decoded,
                                          synthetic_near_miss_invalid_hint) &&
                  near_miss_invalid_hint_decoded.entrypoint_name ==
                      manifest.entrypoint_name &&
                  MatchesUnknownHelperSurface(
                      near_miss_invalid_hint_decoded) &&
                  MatchesTopLevelFlags(near_miss_invalid_hint_decoded,
                                       false,
                                       false,
                                       false,
                                       false) &&
                  MatchesLayout(near_miss_invalid_hint_decoded,
                                ExpectedLayout{}) &&
                  near_miss_invalid_hint_decoded.operand_roles.binding_count ==
                      0 &&
                  near_miss_invalid_hint_decoded.operand_slots.binding_count ==
                      0 &&
                  near_miss_invalid_hint_decoded.operand_descriptors
                          .descriptor_count == 0,
              "expected synthetic routed route-info with near-miss known opcode to ignore invalid caller decode-hint while keeping empty unknown structure")) {
        return 1;
      }
    }
  }
  for (const StubDecoderRouteInfo& route_info : GetStubDecoderRouteInfos()) {
    for (const StubDecoderRouteManifest& manifest : GetStubDecoderRouteManifests()) {
      if (manifest.route == route_info.route) {
        continue;
      }
      const StubDecodedInstruction wrong_entrypoint =
          DecodeViaExplicitRouteEntrypoint(manifest.route,
                                          route_info.instruction_name);
      if (!Expect(
              MatchesUnsupportedRouteDecodeForRoutedSeed(wrong_entrypoint,
                                                         route_info),
              "expected wrong route-keyed entrypoint to reject routed seed while preserving route metadata")) {
        return 1;
      }
    }
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
