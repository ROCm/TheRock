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
    case StubOperandLayoutKind::kWmmaF32_16x16x4_F32W32:
      return "kWmmaF32_16x16x4_F32W32";
    case StubOperandLayoutKind::kWmmaLdScalePairedB32:
      return "kWmmaLdScalePairedB32";
    case StubOperandLayoutKind::kTensorLoadToLds:
      return "kTensorLoadToLds";
    case StubOperandLayoutKind::kTensorStoreFromLds:
      return "kTensorStoreFromLds";
    case StubOperandLayoutKind::kUnknown:
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
