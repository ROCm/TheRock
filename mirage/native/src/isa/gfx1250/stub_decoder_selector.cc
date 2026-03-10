#include "lib/sim/isa/gfx1250/stub_decoder_selector.h"

#include <algorithm>
#include <array>
#include <vector>

namespace mirage::sim::isa::gfx1250 {
namespace {

struct RouteSpec {
  StubDecoderRoute route;
  std::string_view route_name;
  std::uint32_t route_priority;
};

constexpr std::array<RouteSpec, 4> kRouteSpecs{{
    {StubDecoderRoute::kVop3p, "kVop3p", 1},
    {StubDecoderRoute::kMimgTensor, "kMimgTensor", 2},
    {StubDecoderRoute::kVop1, "kVop1", 3},
    {StubDecoderRoute::kVop3Sdst, "kVop3Sdst", 4},
}};

constexpr std::size_t RouteIndex(StubDecoderRoute route) {
  switch (route) {
    case StubDecoderRoute::kVop3p:
      return 0;
    case StubDecoderRoute::kMimgTensor:
      return 1;
    case StubDecoderRoute::kVop1:
      return 2;
    case StubDecoderRoute::kVop3Sdst:
      return 3;
    case StubDecoderRoute::kUnsupported:
      break;
  }
  return kRouteSpecs.size();
}

constexpr const RouteSpec* FindRouteSpec(StubDecoderRoute route) {
  for (const RouteSpec& spec : kRouteSpecs) {
    if (spec.route == route) {
      return &spec;
    }
  }
  return nullptr;
}

struct BuiltRouteTables {
  std::vector<StubDecoderRouteInfo> route_infos;
  std::array<std::vector<std::string_view>, kRouteSpecs.size()> route_instructions;
  std::array<StubDecoderRouteManifest, kRouteSpecs.size()> manifests;
};

const BuiltRouteTables& GetBuiltRouteTables() {
  static const BuiltRouteTables tables = [] {
    BuiltRouteTables built;

    for (std::size_t i = 0; i < kRouteSpecs.size(); ++i) {
      built.manifests[i] = {
          kRouteSpecs[i].route,
          kRouteSpecs[i].route_name,
          kRouteSpecs[i].route_priority,
          0,
          0,
          0,
          0,
      };
    }

    for (const DecoderSeedInfo& seed : GetDecoderSeedInfos()) {
      const StubDecoderRoute route = SelectStubDecoderRoute(seed.decode_hint);
      const RouteSpec* spec = FindRouteSpec(route);
      if (spec == nullptr) {
        continue;
      }

      built.route_infos.push_back({
          seed.instruction_name,
          spec->route,
          spec->route_name,
          spec->route_priority,
          seed.decode_hint,
          seed.rdna4_encoding_name,
          seed.rdna4_opcode,
          seed.rdna4_operand_count,
          seed.appears_in_rdna4_xml,
          seed.is_target_specific,
      });

      const std::size_t route_index = RouteIndex(route);
      built.route_instructions[route_index].push_back(seed.instruction_name);

      StubDecoderRouteManifest& manifest = built.manifests[route_index];
      ++manifest.instruction_count;
      if (seed.appears_in_rdna4_xml) {
        ++manifest.xml_backed_count;
      } else {
        ++manifest.llvm_only_count;
      }
      if (seed.is_target_specific) {
        ++manifest.target_specific_count;
      }
    }

    std::sort(built.route_infos.begin(), built.route_infos.end(),
              [](const StubDecoderRouteInfo& lhs, const StubDecoderRouteInfo& rhs) {
                if (lhs.route_priority != rhs.route_priority) {
                  return lhs.route_priority < rhs.route_priority;
                }
                return lhs.instruction_name < rhs.instruction_name;
              });
    for (auto& route_values : built.route_instructions) {
      std::sort(route_values.begin(), route_values.end());
    }

    return built;
  }();
  return tables;
}

}  // namespace

StubDecoderRoute SelectStubDecoderRoute(DecodeSeedHint decode_hint) {
  switch (decode_hint) {
    case DecodeSeedHint::kVop3p:
      return StubDecoderRoute::kVop3p;
    case DecodeSeedHint::kMimgTensor:
      return StubDecoderRoute::kMimgTensor;
    case DecodeSeedHint::kVop1:
      return StubDecoderRoute::kVop1;
    case DecodeSeedHint::kVop3Sdst:
      return StubDecoderRoute::kVop3Sdst;
    case DecodeSeedHint::kUnknown:
    case DecodeSeedHint::kVop3:
      return StubDecoderRoute::kUnsupported;
  }
  return StubDecoderRoute::kUnsupported;
}

StubDecoderRoute SelectStubDecoderRoute(std::string_view instruction_name) {
  const DecoderSeedInfo* seed = FindDecoderSeedInfo(instruction_name);
  if (seed == nullptr) {
    return StubDecoderRoute::kUnsupported;
  }
  return SelectStubDecoderRoute(seed->decode_hint);
}

std::string_view GetStubDecoderRouteName(StubDecoderRoute route) {
  const RouteSpec* spec = FindRouteSpec(route);
  if (spec == nullptr) {
    return "kUnsupported";
  }
  return spec->route_name;
}

std::span<const StubDecoderRouteInfo> GetStubDecoderRouteInfos() {
  return GetBuiltRouteTables().route_infos;
}

const StubDecoderRouteInfo* FindStubDecoderRouteInfo(
    std::string_view instruction_name) {
  for (const StubDecoderRouteInfo& info : GetStubDecoderRouteInfos()) {
    if (info.instruction_name == instruction_name) {
      return &info;
    }
  }
  return nullptr;
}

std::span<const std::string_view> GetStubDecoderRouteInstructions(
    StubDecoderRoute route) {
  const std::size_t route_index = RouteIndex(route);
  if (route_index >= kRouteSpecs.size()) {
    return std::span<const std::string_view>();
  }
  return GetBuiltRouteTables().route_instructions[route_index];
}

std::span<const StubDecoderRouteManifest> GetStubDecoderRouteManifests() {
  return GetBuiltRouteTables().manifests;
}

const StubDecoderRouteManifest* FindStubDecoderRouteManifest(
    StubDecoderRoute route) {
  const std::size_t route_index = RouteIndex(route);
  if (route_index >= kRouteSpecs.size()) {
    return nullptr;
  }
  return &GetBuiltRouteTables().manifests[route_index];
}

}  // namespace mirage::sim::isa::gfx1250
