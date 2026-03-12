#include "lib/sim/isa/gfx1250/stub_decoder.h"

#include <array>

namespace mirage::sim::isa::gfx1250 {
namespace {

struct RouteEntrypointSpec {
  StubDecoderRoute route;
  std::string_view entrypoint_name;
};

struct ClassifiedStubShape {
  StubOpcodeShape opcode_shape = StubOpcodeShape::kUnknown;
  StubExecutionDomain execution_domain = StubExecutionDomain::kUnknown;
  bool uses_accumulator = false;
  bool uses_tensor_memory = false;
  bool uses_scale_path = false;
  bool uses_paired_operands = false;
};

struct ParsedMatrixInstructionShape {
  bool valid = false;
  std::uint16_t rows = 0;
  std::uint16_t columns = 0;
  std::uint16_t depth = 0;
  std::uint8_t input_element_bit_width = 0;
  std::uint8_t result_element_bit_width = 0;
  std::uint8_t wave_size = 0;
};

StubOperandRoleRecord MakeOperandRoles(
    std::initializer_list<StubOperandRoleBinding> bindings) {
  StubOperandRoleRecord record;
  std::uint32_t index = 0;
  for (const StubOperandRoleBinding& binding : bindings) {
    if (index >= record.bindings.size()) {
      break;
    }
    record.bindings[index++] = binding;
  }
  record.binding_count = index;
  return record;
}

StubOperandSlotRecord MakeOperandSlots(
    std::initializer_list<StubOperandSlotBinding> bindings) {
  StubOperandSlotRecord record;
  std::uint32_t index = 0;
  for (const StubOperandSlotBinding& binding : bindings) {
    if (index >= record.bindings.size()) {
      break;
    }
    record.bindings[index++] = binding;
  }
  record.binding_count = index;
  return record;
}

bool HasPrefix(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() &&
         value.substr(0, prefix.size()) == prefix;
}

bool ContainsToken(std::string_view value, std::string_view token) {
  return value.find(token) != std::string_view::npos;
}

bool ParseUnsigned16(std::string_view token, std::uint16_t* out) {
  if (token.empty()) {
    return false;
  }
  std::uint32_t parsed = 0;
  for (char ch : token) {
    if (ch < '0' || ch > '9') {
      return false;
    }
    parsed = parsed * 10u + static_cast<std::uint32_t>(ch - '0');
  }
  *out = static_cast<std::uint16_t>(parsed);
  return true;
}

bool ParseDimensionTriplet(std::string_view instruction_name,
                           std::uint16_t* rows,
                           std::uint16_t* columns,
                           std::uint16_t* depth) {
  for (std::size_t start = 0; start < instruction_name.size(); ++start) {
    if (instruction_name[start] < '0' || instruction_name[start] > '9') {
      continue;
    }
    const std::size_t first_x = instruction_name.find('X', start);
    if (first_x == std::string_view::npos) {
      continue;
    }
    const std::size_t second_x = instruction_name.find('X', first_x + 1);
    if (second_x == std::string_view::npos) {
      continue;
    }
    const std::size_t end =
        instruction_name.find('_', second_x + 1) == std::string_view::npos
            ? instruction_name.size()
            : instruction_name.find('_', second_x + 1);
    std::uint16_t parsed_rows = 0;
    std::uint16_t parsed_columns = 0;
    std::uint16_t parsed_depth = 0;
    if (!ParseUnsigned16(instruction_name.substr(start, first_x - start),
                         &parsed_rows) ||
        !ParseUnsigned16(
            instruction_name.substr(first_x + 1, second_x - first_x - 1),
            &parsed_columns) ||
        !ParseUnsigned16(instruction_name.substr(second_x + 1, end - second_x - 1),
                         &parsed_depth)) {
      continue;
    }
    *rows = parsed_rows;
    *columns = parsed_columns;
    *depth = parsed_depth;
    return true;
  }
  return false;
}

std::uint8_t InferMatrixResultWidth(std::string_view instruction_name) {
  if (HasPrefix(instruction_name, "V_WMMA_F32_") ||
      HasPrefix(instruction_name, "V_SWMMAC_F32_") ||
      HasPrefix(instruction_name, "V_WMMA_SCALE_F32_") ||
      HasPrefix(instruction_name, "V_WMMA_SCALE16_F32_") ||
      HasPrefix(instruction_name, "V_WMMA_BF16F32_") ||
      HasPrefix(instruction_name, "V_SWMMAC_BF16F32_") ||
      HasPrefix(instruction_name, "V_WMMA_I32_") ||
      HasPrefix(instruction_name, "V_SWMMAC_I32_")) {
    return 32;
  }
  if (HasPrefix(instruction_name, "V_WMMA_F16_") ||
      HasPrefix(instruction_name, "V_SWMMAC_F16_") ||
      HasPrefix(instruction_name, "V_WMMA_BF16_") ||
      HasPrefix(instruction_name, "V_SWMMAC_BF16_")) {
    return 16;
  }
  return 0;
}

std::uint8_t InferMatrixInputWidth(std::string_view instruction_name) {
  if (ContainsToken(instruction_name, "_F4_")) {
    return 4;
  }
  if (ContainsToken(instruction_name, "FP8") ||
      ContainsToken(instruction_name, "BF8") ||
      ContainsToken(instruction_name, "IU8") ||
      ContainsToken(instruction_name, "F8F6F4")) {
    return 8;
  }
  if (ContainsToken(instruction_name, "BF16") ||
      ContainsToken(instruction_name, "_F16_")) {
    return 16;
  }
  if (ContainsToken(instruction_name, "_F32_") &&
      ContainsToken(instruction_name, "16X16X4")) {
    return 32;
  }
  return 0;
}

std::uint8_t InferWaveSize(std::string_view instruction_name) {
  const std::size_t wave_marker = instruction_name.rfind("_w");
  if (wave_marker == std::string_view::npos) {
    // gfx1250 is wave32 in Mirage. LLVM-style routed seeds may omit the
    // explicit `_w32` suffix, but the local stub layer still materializes
    // matrix fragments as wave32.
    return 32;
  }
  std::uint16_t parsed_wave = 0;
  if (!ParseUnsigned16(instruction_name.substr(wave_marker + 2), &parsed_wave)) {
    return 0;
  }
  return parsed_wave == 32 ? 32 : 0;
}

ParsedMatrixInstructionShape ParseMatrixInstructionShape(
    std::string_view instruction_name) {
  if (!HasPrefix(instruction_name, "V_WMMA_") &&
      !HasPrefix(instruction_name, "V_SWMMAC_")) {
    return {};
  }

  ParsedMatrixInstructionShape parsed;
  parsed.valid = ParseDimensionTriplet(instruction_name, &parsed.rows,
                                       &parsed.columns, &parsed.depth);
  if (!parsed.valid) {
    return {};
  }
  parsed.input_element_bit_width = InferMatrixInputWidth(instruction_name);
  parsed.result_element_bit_width = InferMatrixResultWidth(instruction_name);
  parsed.wave_size = InferWaveSize(instruction_name);
  parsed.valid = parsed.input_element_bit_width != 0 &&
                 parsed.result_element_bit_width != 0 &&
                 parsed.wave_size != 0;
  return parsed;
}

FragmentShape ClassifyOperandFragmentShape(
    std::string_view instruction_name,
    const StubOperandSlotBinding& binding) {
  const bool is_result_slot =
      binding.slot_kind == StubOperandSlotKind::kDestination ||
      binding.slot_kind == StubOperandSlotKind::kScalarDestination;
  switch (binding.value_class) {
    case StubOperandValueClass::kPackedVector:
      if (instruction_name == "V_CVT_PK_F16_FP8" ||
          instruction_name == "V_CVT_PK_F16_BF8") {
        if (binding.slot_kind == StubOperandSlotKind::kSource0) {
          return MakePackedFragmentShape(2, 8);
        }
        return MakePackedFragmentShape(2, 16);
      }
      return MakePackedFragmentShape(binding.component_count, 16);
    case StubOperandValueClass::kVectorRegister:
      if (instruction_name == "V_CVT_F16_FP8" ||
          instruction_name == "V_CVT_F16_BF8") {
        return MakeScalarFragmentShape(is_result_slot ? 16 : 8);
      }
      if (instruction_name == "V_CVT_F32_FP8") {
        return MakeScalarFragmentShape(is_result_slot ? 32 : 8);
      }
      if (instruction_name == "V_DIV_SCALE_F64") {
        return MakeScalarFragmentShape(64);
      }
      if (instruction_name == "V_WMMA_LD_SCALE16_PAIRED_B64") {
        return MakeVectorFragmentShape(binding.component_count, 64);
      }
      return MakeVectorFragmentShape(binding.component_count, 32);
    case StubOperandValueClass::kScalarRegister:
      return MakeScalarFragmentShape(32);
    case StubOperandValueClass::kTensorDescriptor:
      return MakeTensorDescriptorFragmentShape();
    case StubOperandValueClass::kTensorCoordinate:
      return MakeTensorCoordinateFragmentShape();
    case StubOperandValueClass::kLdsAddress:
      return MakeAddressFragmentShape(32);
    case StubOperandValueClass::kMatrixFragment:
    case StubOperandValueClass::kAccumulatorFragment:
      if (const ParsedMatrixInstructionShape parsed =
              ParseMatrixInstructionShape(instruction_name);
          parsed.valid) {
        const bool wide_result =
            binding.value_class == StubOperandValueClass::kAccumulatorFragment ||
            is_result_slot;
        return MakeMatrixFragmentShape(
            parsed.rows, parsed.columns, parsed.depth,
            wide_result ? parsed.result_element_bit_width
                        : parsed.input_element_bit_width,
            parsed.wave_size);
      }
      break;
    case StubOperandValueClass::kUnknown:
      break;
  }
  return {};
}

StubOperandSlotRecord AttachFragmentShapes(
    std::string_view instruction_name,
    StubOperandSlotRecord record) {
  for (std::uint32_t index = 0; index < record.binding_count; ++index) {
    record.bindings[index].fragment_shape =
        ClassifyOperandFragmentShape(instruction_name, record.bindings[index]);
  }
  return record;
}

StubOperandRole RoleFromSlotKind(StubOperandSlotKind slot_kind) {
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

StubOperandAccess AccessFromSlotKind(StubOperandSlotKind slot_kind) {
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

StubOperandDescriptorRecord BuildOperandDescriptors(
    const StubOperandSlotRecord& operand_slots) {
  StubOperandDescriptorRecord record;
  record.descriptor_count = operand_slots.binding_count;
  for (std::uint32_t index = 0; index < operand_slots.binding_count; ++index) {
    const StubOperandSlotBinding& slot = operand_slots.bindings[index];
    record.descriptors[index] = {
        RoleFromSlotKind(slot.slot_kind),
        slot.slot_kind,
        slot.value_class,
        AccessFromSlotKind(slot.slot_kind),
        slot.fragment_shape,
        static_cast<std::uint8_t>(slot.component_count),
        slot.is_implicit,
    };
  }
  return record;
}

StubOperandLayoutRecord ClassifyOperandLayout(std::string_view instruction_name) {
  if (instruction_name == "V_PK_ADD_BF16") {
    return {
        StubOperandLayoutKind::kPkAddBf16,
        2,
        1,
        0,
        false,
        false,
        false,
        false,
        false,
    };
  }
  if (instruction_name == "V_PK_FMA_BF16") {
    return {
        StubOperandLayoutKind::kPkFmaBf16,
        3,
        1,
        0,
        false,
        false,
        false,
        false,
        false,
    };
  }
  if (instruction_name == "V_PK_MUL_BF16") {
    return {
        StubOperandLayoutKind::kPkMulBf16,
        2,
        1,
        0,
        false,
        false,
        false,
        false,
        false,
    };
  }
  if (instruction_name == "V_PK_MIN_NUM_BF16") {
    return {
        StubOperandLayoutKind::kPkMinNumBf16,
        2,
        1,
        0,
        false,
        false,
        false,
        false,
        false,
    };
  }
  if (instruction_name == "V_PK_MAX_NUM_BF16") {
    return {
        StubOperandLayoutKind::kPkMaxNumBf16,
        2,
        1,
        0,
        false,
        false,
        false,
        false,
        false,
    };
  }
  if (instruction_name == "V_WMMA_F32_16X16X4_F32_w32") {
    return {
        StubOperandLayoutKind::kWmmaF32_16x16x4_F32W32,
        2,
        1,
        1,
        false,
        false,
        false,
        false,
        false,
    };
  }
  if (instruction_name == "V_WMMA_F32_16X16X128_FP8_FP8_w32") {
    return {
        StubOperandLayoutKind::kWmmaF32_16x16x128_Fp8Fp8W32,
        2,
        1,
        1,
        false,
        false,
        false,
        false,
        false,
    };
  }
  if (instruction_name == "V_WMMA_F16_16X16X128_FP8_FP8_w32") {
    return {
        StubOperandLayoutKind::kWmmaF16_16x16x128_Fp8Fp8W32,
        2,
        1,
        1,
        false,
        false,
        false,
        false,
        false,
    };
  }
  if (instruction_name == "V_WMMA_F32_16X16X64_FP8_FP8_w32") {
    return {
        StubOperandLayoutKind::kWmmaF32_16x16x64_Fp8Fp8W32,
        2,
        1,
        1,
        false,
        false,
        false,
        false,
        false,
    };
  }
  if (instruction_name == "V_WMMA_SCALE_F32_16X16X128_F8F6F4") {
    return {
        StubOperandLayoutKind::kWmmaScaleF32_16x16x128_F8F6F4,
        3,
        1,
        1,
        true,
        false,
        false,
        false,
        false,
    };
  }
  if (instruction_name == "V_WMMA_SCALE16_F32_16X16X128_F8F6F4") {
    return {
        StubOperandLayoutKind::kWmmaScale16F32_16x16x128_F8F6F4,
        3,
        1,
        1,
        true,
        false,
        false,
        false,
        false,
    };
  }
  if (instruction_name == "V_WMMA_LD_SCALE_PAIRED_B32") {
    return {
        StubOperandLayoutKind::kWmmaLdScalePairedB32,
        2,
        1,
        0,
        true,
        true,
        false,
        false,
        false,
    };
  }
  if (instruction_name == "V_WMMA_LD_SCALE16_PAIRED_B64") {
    return {
        StubOperandLayoutKind::kWmmaLdScale16PairedB64,
        2,
        1,
        0,
        true,
        true,
        false,
        false,
        false,
    };
  }
  if (instruction_name == "V_SWMMAC_F32_16X16X128_FP8_FP8_w32") {
    return {
        StubOperandLayoutKind::kSwmmacF32_16x16x128_Fp8Fp8W32,
        2,
        1,
        1,
        false,
        false,
        false,
        false,
        false,
    };
  }
  if (instruction_name == "V_SWMMAC_F16_16X16X128_FP8_FP8_w32") {
    return {
        StubOperandLayoutKind::kSwmmacF16_16x16x128_Fp8Fp8W32,
        2,
        1,
        1,
        false,
        false,
        false,
        false,
        false,
    };
  }
  if (HasPrefix(instruction_name, "V_WMMA_SCALE") &&
      !HasPrefix(instruction_name, "V_WMMA_LD_SCALE")) {
    return {
        StubOperandLayoutKind::kWmmaScaleGeneric,
        3,
        1,
        1,
        true,
        false,
        false,
        false,
        false,
    };
  }
  if (HasPrefix(instruction_name, "V_WMMA_") &&
      !HasPrefix(instruction_name, "V_WMMA_SCALE") &&
      !HasPrefix(instruction_name, "V_WMMA_LD_SCALE")) {
    return {
        StubOperandLayoutKind::kWmmaCoreGeneric,
        2,
        1,
        1,
        false,
        false,
        false,
        false,
        false,
    };
  }
  if (HasPrefix(instruction_name, "V_SWMMAC_")) {
    return {
        StubOperandLayoutKind::kSwmmacCoreGeneric,
        2,
        1,
        1,
        false,
        false,
        false,
        false,
        false,
    };
  }
  if (instruction_name == "TENSOR_LOAD_TO_LDS") {
    return {
        StubOperandLayoutKind::kTensorLoadToLds,
        2,
        0,
        0,
        false,
        false,
        true,
        true,
        false,
    };
  }
  if (instruction_name == "TENSOR_STORE_FROM_LDS") {
    return {
        StubOperandLayoutKind::kTensorStoreFromLds,
        2,
        0,
        0,
        false,
        false,
        true,
        true,
        true,
    };
  }
  if (instruction_name == "V_CVT_F16_BF8") {
    return {
        StubOperandLayoutKind::kCvtF16Bf8,
        1,
        1,
        0,
        false,
        false,
        false,
        false,
        false,
    };
  }
  if (instruction_name == "V_CVT_F16_FP8") {
    return {
        StubOperandLayoutKind::kCvtF16Fp8,
        1,
        1,
        0,
        false,
        false,
        false,
        false,
        false,
    };
  }
  if (instruction_name == "V_CVT_F32_FP8") {
    return {
        StubOperandLayoutKind::kCvtF32Fp8,
        1,
        1,
        0,
        false,
        false,
        false,
        false,
        false,
    };
  }
  if (instruction_name == "V_CVT_PK_F16_FP8") {
    return {
        StubOperandLayoutKind::kCvtPkF16Fp8,
        1,
        1,
        0,
        false,
        false,
        false,
        false,
        false,
    };
  }
  if (instruction_name == "V_CVT_PK_F16_BF8") {
    return {
        StubOperandLayoutKind::kCvtPkF16Bf8,
        1,
        1,
        0,
        false,
        false,
        false,
        false,
        false,
    };
  }
  if (instruction_name == "V_DIV_SCALE_F64") {
    return {
        StubOperandLayoutKind::kVDivScaleF64,
        3,
        2,
        0,
        true,
        false,
        false,
        false,
        false,
    };
  }
  return {};
}

constexpr std::array<RouteEntrypointSpec, 4> kRouteEntrypointSpecs{{
    {StubDecoderRoute::kVop3p, "DecodeVop3pStub"},
    {StubDecoderRoute::kMimgTensor, "DecodeMimgTensorStub"},
    {StubDecoderRoute::kVop1, "DecodeVop1Stub"},
    {StubDecoderRoute::kVop3Sdst, "DecodeVop3SdstStub"},
}};

constexpr std::string_view FindEntrypointName(StubDecoderRoute route) {
  for (const RouteEntrypointSpec& spec : kRouteEntrypointSpecs) {
    if (spec.route == route) {
      return spec.entrypoint_name;
    }
  }
  return "DecodeUnsupportedStub";
}

bool StartsWith(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() &&
         value.substr(0, prefix.size()) == prefix;
}

ClassifiedStubShape ClassifyVop3pShape(std::string_view instruction_name) {
  if (StartsWith(instruction_name, "V_WMMA_LD_SCALE")) {
    return {
        StubOpcodeShape::kWmmaScalePairedLoad,
        StubExecutionDomain::kMatrix,
        false,
        false,
        true,
        true,
    };
  }
  if (StartsWith(instruction_name, "V_WMMA_SCALE")) {
    return {
        StubOpcodeShape::kWmmaScale,
        StubExecutionDomain::kMatrix,
        true,
        false,
        true,
        false,
    };
  }
  if (StartsWith(instruction_name, "V_WMMA_")) {
    return {
        StubOpcodeShape::kWmmaCore,
        StubExecutionDomain::kMatrix,
        true,
        false,
        false,
        false,
    };
  }
  if (StartsWith(instruction_name, "V_SWMMAC_")) {
    return {
        StubOpcodeShape::kSwmmacCore,
        StubExecutionDomain::kMatrix,
        true,
        false,
        false,
        false,
    };
  }
  if (StartsWith(instruction_name, "V_PK_FMA")) {
    return {
        StubOpcodeShape::kVop3pPackedFma,
        StubExecutionDomain::kVectorAlu,
        false,
        false,
        false,
        true,
    };
  }
  if (StartsWith(instruction_name, "V_PK_")) {
    return {
        StubOpcodeShape::kVop3pPackedBinary,
        StubExecutionDomain::kVectorAlu,
        false,
        false,
        false,
        true,
    };
  }
  return {};
}

ClassifiedStubShape ClassifyMimgTensorShape(std::string_view instruction_name) {
  if (instruction_name == "TENSOR_LOAD_TO_LDS") {
    return {
        StubOpcodeShape::kTensorLoadToLds,
        StubExecutionDomain::kTensorMemory,
        false,
        true,
        false,
        false,
    };
  }
  if (instruction_name == "TENSOR_STORE_FROM_LDS") {
    return {
        StubOpcodeShape::kTensorStoreFromLds,
        StubExecutionDomain::kTensorMemory,
        false,
        true,
        false,
        false,
    };
  }
  return {};
}

ClassifiedStubShape ClassifyVop1Shape(std::string_view instruction_name) {
  if (StartsWith(instruction_name, "V_CVT_F16_")) {
    return {
        StubOpcodeShape::kFp8ConvertToF16,
        StubExecutionDomain::kConversion,
        false,
        false,
        false,
        false,
    };
  }
  if (StartsWith(instruction_name, "V_CVT_F32_")) {
    return {
        StubOpcodeShape::kFp8ConvertToF32,
        StubExecutionDomain::kConversion,
        false,
        false,
        false,
        false,
    };
  }
  if (StartsWith(instruction_name, "V_CVT_PK_")) {
    return {
        StubOpcodeShape::kFp8PackedConvert,
        StubExecutionDomain::kConversion,
        false,
        false,
        false,
        true,
    };
  }
  return {};
}

ClassifiedStubShape ClassifyVop3SdstShape(std::string_view instruction_name) {
  if (instruction_name == "V_DIV_SCALE_F64") {
    return {
        StubOpcodeShape::kVop3SdstScale,
        StubExecutionDomain::kScaleAssist,
        false,
        false,
        true,
        false,
    };
  }
  return {};
}

StubOperandRoleRecord ClassifyOperandRoles(std::string_view instruction_name) {
  if (instruction_name == "V_PK_ADD_BF16") {
    return MakeOperandRoles({
        {StubOperandRole::kSource0, 1, false, false},
        {StubOperandRole::kSource1, 1, false, false},
        {StubOperandRole::kDestination, 1, true, false},
    });
  }
  if (instruction_name == "V_PK_FMA_BF16") {
    return MakeOperandRoles({
        {StubOperandRole::kSource0, 1, false, false},
        {StubOperandRole::kSource1, 1, false, false},
        {StubOperandRole::kSource2, 1, false, false},
        {StubOperandRole::kDestination, 1, true, false},
    });
  }
  if (instruction_name == "V_PK_MUL_BF16" ||
      instruction_name == "V_PK_MIN_NUM_BF16" ||
      instruction_name == "V_PK_MAX_NUM_BF16") {
    return MakeOperandRoles({
        {StubOperandRole::kSource0, 1, false, false},
        {StubOperandRole::kSource1, 1, false, false},
        {StubOperandRole::kDestination, 1, true, false},
    });
  }
  if (instruction_name == "V_WMMA_F32_16X16X4_F32_w32" ||
      instruction_name == "V_WMMA_F32_16X16X128_FP8_FP8_w32" ||
      instruction_name == "V_WMMA_F16_16X16X128_FP8_FP8_w32" ||
      instruction_name == "V_WMMA_F32_16X16X64_FP8_FP8_w32") {
    return MakeOperandRoles({
        {StubOperandRole::kSource0, 1, false, false},
        {StubOperandRole::kSource1, 1, false, false},
        {StubOperandRole::kAccumulator, 1, false, false},
        {StubOperandRole::kDestination, 1, true, false},
    });
  }
  if (instruction_name == "V_WMMA_SCALE_F32_16X16X128_F8F6F4" ||
      instruction_name == "V_WMMA_SCALE16_F32_16X16X128_F8F6F4") {
    return MakeOperandRoles({
        {StubOperandRole::kSource0, 1, false, false},
        {StubOperandRole::kSource1, 1, false, false},
        {StubOperandRole::kAccumulator, 1, false, false},
        {StubOperandRole::kScale, 1, false, false},
        {StubOperandRole::kDestination, 1, true, false},
    });
  }
  if (instruction_name == "V_WMMA_LD_SCALE_PAIRED_B32") {
    return MakeOperandRoles({
        {StubOperandRole::kSource0, 1, false, false},
        {StubOperandRole::kScale, 1, false, false},
        {StubOperandRole::kPairedScale, 1, false, false},
        {StubOperandRole::kDestination, 1, true, false},
    });
  }
  if (instruction_name == "V_WMMA_LD_SCALE16_PAIRED_B64") {
    return MakeOperandRoles({
        {StubOperandRole::kSource0, 1, false, false},
        {StubOperandRole::kScale, 1, false, false},
        {StubOperandRole::kPairedScale, 1, false, false},
        {StubOperandRole::kDestination, 1, true, false},
    });
  }
  if (instruction_name == "V_SWMMAC_F32_16X16X128_FP8_FP8_w32" ||
      instruction_name == "V_SWMMAC_F16_16X16X128_FP8_FP8_w32") {
    return MakeOperandRoles({
        {StubOperandRole::kSource0, 1, false, false},
        {StubOperandRole::kSource1, 1, false, false},
        {StubOperandRole::kAccumulator, 1, false, false},
        {StubOperandRole::kDestination, 1, true, false},
    });
  }
  if (HasPrefix(instruction_name, "V_WMMA_SCALE") &&
      !HasPrefix(instruction_name, "V_WMMA_LD_SCALE")) {
    return MakeOperandRoles({
        {StubOperandRole::kSource0, 1, false, false},
        {StubOperandRole::kSource1, 1, false, false},
        {StubOperandRole::kAccumulator, 1, false, false},
        {StubOperandRole::kScale, 1, false, false},
        {StubOperandRole::kDestination, 1, true, false},
    });
  }
  if (HasPrefix(instruction_name, "V_WMMA_") &&
      !HasPrefix(instruction_name, "V_WMMA_SCALE") &&
      !HasPrefix(instruction_name, "V_WMMA_LD_SCALE")) {
    return MakeOperandRoles({
        {StubOperandRole::kSource0, 1, false, false},
        {StubOperandRole::kSource1, 1, false, false},
        {StubOperandRole::kAccumulator, 1, false, false},
        {StubOperandRole::kDestination, 1, true, false},
    });
  }
  if (HasPrefix(instruction_name, "V_SWMMAC_")) {
    return MakeOperandRoles({
        {StubOperandRole::kSource0, 1, false, false},
        {StubOperandRole::kSource1, 1, false, false},
        {StubOperandRole::kAccumulator, 1, false, false},
        {StubOperandRole::kDestination, 1, true, false},
    });
  }
  if (instruction_name == "TENSOR_LOAD_TO_LDS") {
    return MakeOperandRoles({
        {StubOperandRole::kTensorDescriptor, 1, false, false},
        {StubOperandRole::kTensorCoordinate, 1, false, false},
        {StubOperandRole::kLdsDestination, 1, true, false},
    });
  }
  if (instruction_name == "TENSOR_STORE_FROM_LDS") {
    return MakeOperandRoles({
        {StubOperandRole::kTensorDescriptor, 1, false, false},
        {StubOperandRole::kTensorCoordinate, 1, false, false},
        {StubOperandRole::kLdsSource, 1, false, false},
    });
  }
  if (instruction_name == "V_CVT_F16_FP8") {
    return MakeOperandRoles({
        {StubOperandRole::kSource0, 1, false, false},
        {StubOperandRole::kDestination, 1, true, false},
    });
  }
  if (instruction_name == "V_CVT_F16_BF8") {
    return MakeOperandRoles({
        {StubOperandRole::kSource0, 1, false, false},
        {StubOperandRole::kDestination, 1, true, false},
    });
  }
  if (instruction_name == "V_CVT_F32_FP8") {
    return MakeOperandRoles({
        {StubOperandRole::kSource0, 1, false, false},
        {StubOperandRole::kDestination, 1, true, false},
    });
  }
  if (instruction_name == "V_CVT_PK_F16_FP8" ||
      instruction_name == "V_CVT_PK_F16_BF8") {
    return MakeOperandRoles({
        {StubOperandRole::kSource0, 1, false, false},
        {StubOperandRole::kDestination, 1, true, false},
    });
  }
  if (instruction_name == "V_DIV_SCALE_F64") {
    return MakeOperandRoles({
        {StubOperandRole::kSource0, 1, false, false},
        {StubOperandRole::kSource1, 1, false, false},
        {StubOperandRole::kScale, 1, false, false},
        {StubOperandRole::kDestination, 1, true, false},
    });
  }
  return {};
}

StubOperandSlotRecord ClassifyOperandSlots(std::string_view instruction_name) {
  if (instruction_name == "V_PK_ADD_BF16" ||
      instruction_name == "V_PK_MUL_BF16" ||
      instruction_name == "V_PK_MIN_NUM_BF16" ||
      instruction_name == "V_PK_MAX_NUM_BF16") {
    return AttachFragmentShapes(instruction_name, MakeOperandSlots({
        {StubOperandSlotKind::kDestination,
         StubOperandValueClass::kPackedVector,
         0,
         2,
         true,
         false},
        {StubOperandSlotKind::kSource0,
         StubOperandValueClass::kPackedVector,
         1,
         2,
         false,
         false},
        {StubOperandSlotKind::kSource1,
         StubOperandValueClass::kPackedVector,
         2,
         2,
         false,
         false},
    }));
  }
  if (instruction_name == "V_PK_FMA_BF16") {
    return AttachFragmentShapes(instruction_name, MakeOperandSlots({
        {StubOperandSlotKind::kDestination,
         StubOperandValueClass::kPackedVector,
         0,
         2,
         true,
         false},
        {StubOperandSlotKind::kSource0,
         StubOperandValueClass::kPackedVector,
         1,
         2,
         false,
         false},
        {StubOperandSlotKind::kSource1,
         StubOperandValueClass::kPackedVector,
         2,
         2,
         false,
         false},
        {StubOperandSlotKind::kSource2,
         StubOperandValueClass::kPackedVector,
         3,
         2,
         false,
         false},
    }));
  }
  if (instruction_name == "V_WMMA_F32_16X16X4_F32_w32" ||
      instruction_name == "V_WMMA_F32_16X16X128_FP8_FP8_w32" ||
      instruction_name == "V_WMMA_F16_16X16X128_FP8_FP8_w32" ||
      instruction_name == "V_WMMA_F32_16X16X64_FP8_FP8_w32" ||
      instruction_name == "V_SWMMAC_F32_16X16X128_FP8_FP8_w32" ||
      instruction_name == "V_SWMMAC_F16_16X16X128_FP8_FP8_w32") {
    return AttachFragmentShapes(instruction_name, MakeOperandSlots({
        {StubOperandSlotKind::kDestination,
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
    }));
  }
  if (instruction_name == "V_WMMA_SCALE_F32_16X16X128_F8F6F4" ||
      instruction_name == "V_WMMA_SCALE16_F32_16X16X128_F8F6F4") {
    return AttachFragmentShapes(instruction_name, MakeOperandSlots({
        {StubOperandSlotKind::kDestination,
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
         false},
    }));
  }
  if (instruction_name == "V_WMMA_LD_SCALE_PAIRED_B32" ||
      instruction_name == "V_WMMA_LD_SCALE16_PAIRED_B64") {
    return AttachFragmentShapes(instruction_name, MakeOperandSlots({
        {StubOperandSlotKind::kDestination,
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
         false},
    }));
  }
  if (HasPrefix(instruction_name, "V_WMMA_SCALE") &&
      !HasPrefix(instruction_name, "V_WMMA_LD_SCALE")) {
    return AttachFragmentShapes(instruction_name, MakeOperandSlots({
        {StubOperandSlotKind::kDestination,
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
         false},
    }));
  }
  if (HasPrefix(instruction_name, "V_WMMA_") &&
      !HasPrefix(instruction_name, "V_WMMA_SCALE") &&
      !HasPrefix(instruction_name, "V_WMMA_LD_SCALE")) {
    return AttachFragmentShapes(instruction_name, MakeOperandSlots({
        {StubOperandSlotKind::kDestination,
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
    }));
  }
  if (HasPrefix(instruction_name, "V_SWMMAC_")) {
    return AttachFragmentShapes(instruction_name, MakeOperandSlots({
        {StubOperandSlotKind::kDestination,
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
    }));
  }
  if (instruction_name == "TENSOR_LOAD_TO_LDS") {
    return AttachFragmentShapes(instruction_name, MakeOperandSlots({
        {StubOperandSlotKind::kTensorDescriptorSource,
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
         false},
    }));
  }
  if (instruction_name == "TENSOR_STORE_FROM_LDS") {
    return AttachFragmentShapes(instruction_name, MakeOperandSlots({
        {StubOperandSlotKind::kTensorDescriptorSource,
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
         false},
    }));
  }
  if (instruction_name == "V_CVT_F16_FP8" ||
      instruction_name == "V_CVT_F16_BF8" ||
      instruction_name == "V_CVT_F32_FP8") {
    return AttachFragmentShapes(instruction_name, MakeOperandSlots({
        {StubOperandSlotKind::kDestination,
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
    }));
  }
  if (instruction_name == "V_CVT_PK_F16_FP8" ||
      instruction_name == "V_CVT_PK_F16_BF8") {
    return AttachFragmentShapes(instruction_name, MakeOperandSlots({
        {StubOperandSlotKind::kDestination,
         StubOperandValueClass::kPackedVector,
         0,
         2,
         true,
         false},
        {StubOperandSlotKind::kSource0,
         StubOperandValueClass::kPackedVector,
         1,
         2,
         false,
         false},
    }));
  }
  if (instruction_name == "V_DIV_SCALE_F64") {
    return AttachFragmentShapes(instruction_name, MakeOperandSlots({
        {StubOperandSlotKind::kDestination,
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
         false},
    }));
  }
  return {};
}

ClassifiedStubShape ClassifyStubShape(
    StubDecoderRoute route,
    std::string_view instruction_name) {
  switch (route) {
    case StubDecoderRoute::kVop3p:
      return ClassifyVop3pShape(instruction_name);
    case StubDecoderRoute::kMimgTensor:
      return ClassifyMimgTensorShape(instruction_name);
    case StubDecoderRoute::kVop1:
      return ClassifyVop1Shape(instruction_name);
    case StubDecoderRoute::kVop3Sdst:
      return ClassifyVop3SdstShape(instruction_name);
    case StubDecoderRoute::kUnsupported:
      break;
  }
  return {};
}

StubDecodedInstruction BuildDecodedStub(
    const StubDecoderRouteInfo& route_info,
    std::string_view entrypoint_name) {
  const ClassifiedStubShape classified_shape =
      ClassifyStubShape(route_info.route, route_info.instruction_name);
  const StubOperandLayoutRecord operand_layout =
      ClassifyOperandLayout(route_info.instruction_name);
  const StubOperandRoleRecord operand_roles =
      ClassifyOperandRoles(route_info.instruction_name);
  const StubOperandSlotRecord operand_slots =
      ClassifyOperandSlots(route_info.instruction_name);
  const StubOperandDescriptorRecord operand_descriptors =
      BuildOperandDescriptors(operand_slots);
  return {
      route_info.instruction_name,
      StubDecodeStatus::kDecodedStub,
      route_info.route,
      route_info.route_name,
      entrypoint_name,
      route_info.route_priority,
      route_info.rdna4_encoding_name,
      route_info.rdna4_opcode,
      route_info.rdna4_operand_count,
      route_info.appears_in_rdna4_xml,
      route_info.is_target_specific,
      classified_shape.opcode_shape,
      classified_shape.execution_domain,
      classified_shape.uses_accumulator,
      classified_shape.uses_tensor_memory,
      classified_shape.uses_scale_path,
      classified_shape.uses_paired_operands,
      operand_layout,
      operand_roles,
      operand_slots,
      operand_descriptors,
  };
}

StubDecodedInstruction MakeUnsupportedInstruction(
    std::string_view instruction_name,
    StubDecodeStatus status) {
  return {
      instruction_name,
      status,
      StubDecoderRoute::kUnsupported,
      "kUnsupported",
      "DecodeUnsupportedStub",
      0,
      "",
      0,
      0,
      false,
      false,
      StubOpcodeShape::kUnknown,
      StubExecutionDomain::kUnknown,
      false,
      false,
      false,
      false,
      {},
      {},
      {},
      {},
  };
}

StubDecodedInstruction DecodeRouteStub(
    std::string_view instruction_name,
    StubDecoderRoute expected_route) {
  const StubDecoderRouteInfo* route_info =
      FindStubDecoderRouteInfo(instruction_name);
  if (route_info == nullptr) {
    if (FindDecoderSeedInfo(instruction_name) != nullptr) {
      return MakeUnsupportedInstruction(instruction_name,
                                        StubDecodeStatus::kUnsupportedRoute);
    }
    return MakeUnsupportedInstruction(instruction_name,
                                      StubDecodeStatus::kUnknownInstruction);
  }
  if (route_info->route != expected_route) {
    StubDecodedInstruction result = MakeUnsupportedInstruction(
        instruction_name, StubDecodeStatus::kUnsupportedRoute);
    result.route = route_info->route;
    result.route_name = route_info->route_name;
    result.route_priority = route_info->route_priority;
    result.rdna4_encoding_name = route_info->rdna4_encoding_name;
    result.rdna4_opcode = route_info->rdna4_opcode;
    result.rdna4_operand_count = route_info->rdna4_operand_count;
    result.appears_in_rdna4_xml = route_info->appears_in_rdna4_xml;
    result.is_target_specific = route_info->is_target_specific;
    return result;
  }
  return BuildDecodedStub(*route_info, FindEntrypointName(expected_route));
}

}  // namespace

StubDecodedInstruction DecodeStubInstruction(std::string_view instruction_name) {
  const StubDecoderRouteInfo* route_info =
      FindStubDecoderRouteInfo(instruction_name);
  if (route_info == nullptr) {
    if (FindDecoderSeedInfo(instruction_name) != nullptr) {
      return MakeUnsupportedInstruction(instruction_name,
                                        StubDecodeStatus::kUnsupportedRoute);
    }
    return MakeUnsupportedInstruction(instruction_name,
                                      StubDecodeStatus::kUnknownInstruction);
  }
  return DecodeStubInstruction(*route_info);
}

StubDecodedInstruction DecodeStubInstruction(
    const StubDecoderRouteInfo& route_info) {
  switch (route_info.route) {
    case StubDecoderRoute::kVop3p:
      return BuildDecodedStub(route_info, "DecodeVop3pStub");
    case StubDecoderRoute::kMimgTensor:
      return BuildDecodedStub(route_info, "DecodeMimgTensorStub");
    case StubDecoderRoute::kVop1:
      return BuildDecodedStub(route_info, "DecodeVop1Stub");
    case StubDecoderRoute::kVop3Sdst:
      return BuildDecodedStub(route_info, "DecodeVop3SdstStub");
    case StubDecoderRoute::kUnsupported:
      break;
  }
  return MakeUnsupportedInstruction(route_info.instruction_name,
                                    StubDecodeStatus::kUnsupportedRoute);
}

StubDecodedInstruction DecodeVop3pStub(std::string_view instruction_name) {
  return DecodeRouteStub(instruction_name, StubDecoderRoute::kVop3p);
}

StubDecodedInstruction DecodeMimgTensorStub(std::string_view instruction_name) {
  return DecodeRouteStub(instruction_name, StubDecoderRoute::kMimgTensor);
}

StubDecodedInstruction DecodeVop1Stub(std::string_view instruction_name) {
  return DecodeRouteStub(instruction_name, StubDecoderRoute::kVop1);
}

StubDecodedInstruction DecodeVop3SdstStub(std::string_view instruction_name) {
  return DecodeRouteStub(instruction_name, StubDecoderRoute::kVop3Sdst);
}

std::string_view GetStubOpcodeShapeName(StubOpcodeShape opcode_shape) {
  switch (opcode_shape) {
    case StubOpcodeShape::kVop3pPackedBinary:
      return "kVop3pPackedBinary";
    case StubOpcodeShape::kVop3pPackedFma:
      return "kVop3pPackedFma";
    case StubOpcodeShape::kWmmaCore:
      return "kWmmaCore";
    case StubOpcodeShape::kWmmaScale:
      return "kWmmaScale";
    case StubOpcodeShape::kWmmaScalePairedLoad:
      return "kWmmaScalePairedLoad";
    case StubOpcodeShape::kSwmmacCore:
      return "kSwmmacCore";
    case StubOpcodeShape::kTensorLoadToLds:
      return "kTensorLoadToLds";
    case StubOpcodeShape::kTensorStoreFromLds:
      return "kTensorStoreFromLds";
    case StubOpcodeShape::kFp8ConvertToF16:
      return "kFp8ConvertToF16";
    case StubOpcodeShape::kFp8ConvertToF32:
      return "kFp8ConvertToF32";
    case StubOpcodeShape::kFp8PackedConvert:
      return "kFp8PackedConvert";
    case StubOpcodeShape::kVop3SdstScale:
      return "kVop3SdstScale";
    case StubOpcodeShape::kUnknown:
      break;
  }
  return "kUnknown";
}

std::string_view GetStubExecutionDomainName(
    StubExecutionDomain execution_domain) {
  switch (execution_domain) {
    case StubExecutionDomain::kVectorAlu:
      return "kVectorAlu";
    case StubExecutionDomain::kMatrix:
      return "kMatrix";
    case StubExecutionDomain::kTensorMemory:
      return "kTensorMemory";
    case StubExecutionDomain::kConversion:
      return "kConversion";
    case StubExecutionDomain::kScaleAssist:
      return "kScaleAssist";
    case StubExecutionDomain::kUnknown:
      break;
  }
  return "kUnknown";
}

std::string_view GetStubOperandLayoutName(
    StubOperandLayoutKind operand_layout_kind) {
  switch (operand_layout_kind) {
    case StubOperandLayoutKind::kPkAddBf16:
      return "kPkAddBf16";
    case StubOperandLayoutKind::kPkFmaBf16:
      return "kPkFmaBf16";
    case StubOperandLayoutKind::kPkMulBf16:
      return "kPkMulBf16";
    case StubOperandLayoutKind::kPkMinNumBf16:
      return "kPkMinNumBf16";
    case StubOperandLayoutKind::kPkMaxNumBf16:
      return "kPkMaxNumBf16";
    case StubOperandLayoutKind::kWmmaF32_16x16x4_F32W32:
      return "kWmmaF32_16x16x4_F32W32";
    case StubOperandLayoutKind::kWmmaF32_16x16x128_Fp8Fp8W32:
      return "kWmmaF32_16x16x128_Fp8Fp8W32";
    case StubOperandLayoutKind::kWmmaF16_16x16x128_Fp8Fp8W32:
      return "kWmmaF16_16x16x128_Fp8Fp8W32";
    case StubOperandLayoutKind::kWmmaF32_16x16x64_Fp8Fp8W32:
      return "kWmmaF32_16x16x64_Fp8Fp8W32";
    case StubOperandLayoutKind::kWmmaCoreGeneric:
      return "kWmmaCoreGeneric";
    case StubOperandLayoutKind::kWmmaScaleF32_16x16x128_F8F6F4:
      return "kWmmaScaleF32_16x16x128_F8F6F4";
    case StubOperandLayoutKind::kWmmaScale16F32_16x16x128_F8F6F4:
      return "kWmmaScale16F32_16x16x128_F8F6F4";
    case StubOperandLayoutKind::kWmmaScaleGeneric:
      return "kWmmaScaleGeneric";
    case StubOperandLayoutKind::kWmmaLdScalePairedB32:
      return "kWmmaLdScalePairedB32";
    case StubOperandLayoutKind::kWmmaLdScale16PairedB64:
      return "kWmmaLdScale16PairedB64";
    case StubOperandLayoutKind::kSwmmacF32_16x16x128_Fp8Fp8W32:
      return "kSwmmacF32_16x16x128_Fp8Fp8W32";
    case StubOperandLayoutKind::kSwmmacF16_16x16x128_Fp8Fp8W32:
      return "kSwmmacF16_16x16x128_Fp8Fp8W32";
    case StubOperandLayoutKind::kSwmmacCoreGeneric:
      return "kSwmmacCoreGeneric";
    case StubOperandLayoutKind::kTensorLoadToLds:
      return "kTensorLoadToLds";
    case StubOperandLayoutKind::kTensorStoreFromLds:
      return "kTensorStoreFromLds";
    case StubOperandLayoutKind::kCvtF16Bf8:
      return "kCvtF16Bf8";
    case StubOperandLayoutKind::kCvtF16Fp8:
      return "kCvtF16Fp8";
    case StubOperandLayoutKind::kCvtF32Fp8:
      return "kCvtF32Fp8";
    case StubOperandLayoutKind::kCvtPkF16Fp8:
      return "kCvtPkF16Fp8";
    case StubOperandLayoutKind::kCvtPkF16Bf8:
      return "kCvtPkF16Bf8";
    case StubOperandLayoutKind::kVDivScaleF64:
      return "kVDivScaleF64";
    case StubOperandLayoutKind::kUnknown:
      break;
  }
  return "kUnknown";
}

std::string_view GetStubOperandRoleName(StubOperandRole operand_role) {
  switch (operand_role) {
    case StubOperandRole::kDestination:
      return "kDestination";
    case StubOperandRole::kSource0:
      return "kSource0";
    case StubOperandRole::kSource1:
      return "kSource1";
    case StubOperandRole::kSource2:
      return "kSource2";
    case StubOperandRole::kAccumulator:
      return "kAccumulator";
    case StubOperandRole::kScale:
      return "kScale";
    case StubOperandRole::kPairedScale:
      return "kPairedScale";
    case StubOperandRole::kTensorDescriptor:
      return "kTensorDescriptor";
    case StubOperandRole::kTensorCoordinate:
      return "kTensorCoordinate";
    case StubOperandRole::kLdsDestination:
      return "kLdsDestination";
    case StubOperandRole::kLdsSource:
      return "kLdsSource";
    case StubOperandRole::kUnknown:
      break;
  }
  return "kUnknown";
}

std::string_view GetStubOperandSlotKindName(
    StubOperandSlotKind operand_slot_kind) {
  switch (operand_slot_kind) {
    case StubOperandSlotKind::kDestination:
      return "kDestination";
    case StubOperandSlotKind::kScalarDestination:
      return "kScalarDestination";
    case StubOperandSlotKind::kSource0:
      return "kSource0";
    case StubOperandSlotKind::kSource1:
      return "kSource1";
    case StubOperandSlotKind::kSource2:
      return "kSource2";
    case StubOperandSlotKind::kAccumulatorSource:
      return "kAccumulatorSource";
    case StubOperandSlotKind::kScaleSource:
      return "kScaleSource";
    case StubOperandSlotKind::kPairedScaleSource:
      return "kPairedScaleSource";
    case StubOperandSlotKind::kTensorDescriptorSource:
      return "kTensorDescriptorSource";
    case StubOperandSlotKind::kTensorCoordinateSource:
      return "kTensorCoordinateSource";
    case StubOperandSlotKind::kLdsDestination:
      return "kLdsDestination";
    case StubOperandSlotKind::kLdsSource:
      return "kLdsSource";
    case StubOperandSlotKind::kUnknown:
      break;
  }
  return "kUnknown";
}

std::string_view GetStubOperandValueClassName(
    StubOperandValueClass operand_value_class) {
  switch (operand_value_class) {
    case StubOperandValueClass::kVectorRegister:
      return "kVectorRegister";
    case StubOperandValueClass::kScalarRegister:
      return "kScalarRegister";
    case StubOperandValueClass::kPackedVector:
      return "kPackedVector";
    case StubOperandValueClass::kMatrixFragment:
      return "kMatrixFragment";
    case StubOperandValueClass::kAccumulatorFragment:
      return "kAccumulatorFragment";
    case StubOperandValueClass::kTensorDescriptor:
      return "kTensorDescriptor";
    case StubOperandValueClass::kTensorCoordinate:
      return "kTensorCoordinate";
    case StubOperandValueClass::kLdsAddress:
      return "kLdsAddress";
    case StubOperandValueClass::kUnknown:
      break;
  }
  return "kUnknown";
}

std::span<const StubDecoderEntrypointManifest> GetStubDecoderEntrypointManifests() {
  static const std::array<StubDecoderEntrypointManifest, 4> kEntrypointManifests{{
      {
          StubDecoderRoute::kVop3p,
          "kVop3p",
          "DecodeVop3pStub",
          1,
          FindStubDecoderRouteManifest(StubDecoderRoute::kVop3p)->instruction_count,
      },
      {
          StubDecoderRoute::kMimgTensor,
          "kMimgTensor",
          "DecodeMimgTensorStub",
          2,
          FindStubDecoderRouteManifest(StubDecoderRoute::kMimgTensor)->instruction_count,
      },
      {
          StubDecoderRoute::kVop1,
          "kVop1",
          "DecodeVop1Stub",
          3,
          FindStubDecoderRouteManifest(StubDecoderRoute::kVop1)->instruction_count,
      },
      {
          StubDecoderRoute::kVop3Sdst,
          "kVop3Sdst",
          "DecodeVop3SdstStub",
          4,
          FindStubDecoderRouteManifest(StubDecoderRoute::kVop3Sdst)->instruction_count,
      },
  }};
  return kEntrypointManifests;
}

const StubDecoderEntrypointManifest* FindStubDecoderEntrypointManifest(
    StubDecoderRoute route) {
  for (const StubDecoderEntrypointManifest& manifest :
       GetStubDecoderEntrypointManifests()) {
    if (manifest.route == route) {
      return &manifest;
    }
  }
  return nullptr;
}

}  // namespace mirage::sim::isa::gfx1250
