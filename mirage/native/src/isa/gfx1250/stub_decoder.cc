#include "lib/sim/isa/gfx1250/stub_decoder.h"

#include <array>

namespace mirage::sim::isa::gfx1250 {
namespace {

struct RouteEntrypointSpec {
  StubDecoderRoute route;
  std::string_view entrypoint_name;
};

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

StubDecodedInstruction BuildDecodedStub(
    const StubDecoderRouteInfo& route_info,
    std::string_view entrypoint_name) {
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
  };
}

StubDecodedInstruction DecodeRouteStub(
    std::string_view instruction_name,
    StubDecoderRoute expected_route) {
  const StubDecoderRouteInfo* route_info =
      FindStubDecoderRouteInfo(instruction_name);
  if (route_info == nullptr) {
    if (FindDecoderSeedInfo(instruction_name) != nullptr) {
      return {
          instruction_name,
          StubDecodeStatus::kUnsupportedRoute,
          StubDecoderRoute::kUnsupported,
          "kUnsupported",
          "DecodeUnsupportedStub",
          0,
          "",
          0,
          0,
          false,
          false,
      };
    }
    return {
        instruction_name,
        StubDecodeStatus::kUnknownInstruction,
        StubDecoderRoute::kUnsupported,
        "kUnsupported",
        "DecodeUnsupportedStub",
        0,
        "",
        0,
        0,
        false,
        false,
    };
  }
  if (route_info->route != expected_route) {
    return {
        instruction_name,
        StubDecodeStatus::kUnsupportedRoute,
        route_info->route,
        route_info->route_name,
        "DecodeUnsupportedStub",
        route_info->route_priority,
        route_info->rdna4_encoding_name,
        route_info->rdna4_opcode,
        route_info->rdna4_operand_count,
        route_info->appears_in_rdna4_xml,
        route_info->is_target_specific,
    };
  }
  return BuildDecodedStub(*route_info, FindEntrypointName(expected_route));
}

}  // namespace

StubDecodedInstruction DecodeStubInstruction(std::string_view instruction_name) {
  const StubDecoderRouteInfo* route_info =
      FindStubDecoderRouteInfo(instruction_name);
  if (route_info == nullptr) {
    if (FindDecoderSeedInfo(instruction_name) != nullptr) {
      return {
          instruction_name,
          StubDecodeStatus::kUnsupportedRoute,
          StubDecoderRoute::kUnsupported,
          "kUnsupported",
          "DecodeUnsupportedStub",
          0,
          "",
          0,
          0,
          false,
          false,
      };
    }
    return {
        instruction_name,
        StubDecodeStatus::kUnknownInstruction,
        StubDecoderRoute::kUnsupported,
        "kUnsupported",
        "DecodeUnsupportedStub",
        0,
        "",
        0,
        0,
        false,
        false,
    };
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
  return {
      route_info.instruction_name,
      StubDecodeStatus::kUnsupportedRoute,
      StubDecoderRoute::kUnsupported,
      "kUnsupported",
      "DecodeUnsupportedStub",
      0,
      "",
      0,
      0,
      false,
      false,
  };
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
