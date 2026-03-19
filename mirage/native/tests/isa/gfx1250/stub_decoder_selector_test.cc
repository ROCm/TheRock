#include <algorithm>
#include <iostream>
#include <string_view>
#include <vector>

#include "lib/sim/isa/gfx1250/stub_decoder_selector.h"

namespace {

using mirage::sim::isa::gfx1250::DecoderSeedInfo;
using mirage::sim::isa::gfx1250::GetDecoderSeedInfos;
using mirage::sim::isa::gfx1250::FindStubDecoderRouteInfo;
using mirage::sim::isa::gfx1250::FindStubDecoderRouteManifest;
using mirage::sim::isa::gfx1250::GetStubDecoderRouteInfos;
using mirage::sim::isa::gfx1250::GetStubDecoderRouteInstructions;
using mirage::sim::isa::gfx1250::GetStubDecoderRouteManifests;
using mirage::sim::isa::gfx1250::GetStubDecoderRouteName;
using mirage::sim::isa::gfx1250::SelectStubDecoderRoute;
using mirage::sim::isa::gfx1250::StubDecoderRoute;
using mirage::sim::isa::gfx1250::StubDecoderRouteInfo;
using mirage::sim::isa::gfx1250::StubDecoderRouteManifest;

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

bool Contains(StubDecoderRoute route, std::string_view instruction_name) {
  for (const std::string_view candidate : GetStubDecoderRouteInstructions(route)) {
    if (candidate == instruction_name) {
      return true;
    }
  }
  return false;
}

bool ListedInAnyRoute(std::string_view instruction_name) {
  for (const StubDecoderRouteManifest& manifest : GetStubDecoderRouteManifests()) {
    if (Contains(manifest.route, instruction_name)) {
      return true;
    }
  }
  return false;
}

std::uint32_t CountSeededInstructionsForRoute(StubDecoderRoute route) {
  std::uint32_t count = 0;
  for (const DecoderSeedInfo& seed : GetDecoderSeedInfos()) {
    if (SelectStubDecoderRoute(seed.decode_hint) == route) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountXmlBackedSeededInstructionsForRoute(StubDecoderRoute route) {
  std::uint32_t count = 0;
  for (const DecoderSeedInfo& seed : GetDecoderSeedInfos()) {
    if (SelectStubDecoderRoute(seed.decode_hint) == route &&
        seed.appears_in_rdna4_xml) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountLlvmOnlySeededInstructionsForRoute(StubDecoderRoute route) {
  std::uint32_t count = 0;
  for (const DecoderSeedInfo& seed : GetDecoderSeedInfos()) {
    if (SelectStubDecoderRoute(seed.decode_hint) == route &&
        !seed.appears_in_rdna4_xml) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountTargetSpecificSeededInstructionsForRoute(
    StubDecoderRoute route) {
  std::uint32_t count = 0;
  for (const DecoderSeedInfo& seed : GetDecoderSeedInfos()) {
    if (SelectStubDecoderRoute(seed.decode_hint) == route &&
        seed.is_target_specific) {
      ++count;
    }
  }
  return count;
}

bool MatchesSeedCatalogParity(const StubDecoderRouteInfo& route_info,
                              const DecoderSeedInfo& seed) {
  const StubDecoderRoute expected_route = SelectStubDecoderRoute(seed.decode_hint);
  const StubDecoderRouteManifest* manifest =
      FindStubDecoderRouteManifest(expected_route);
  if (manifest == nullptr) {
    return false;
  }
  return route_info.instruction_name == seed.instruction_name &&
         route_info.route == expected_route &&
         route_info.route_name == manifest->route_name &&
         route_info.route_priority == manifest->route_priority &&
         route_info.decode_hint == seed.decode_hint &&
         route_info.rdna4_encoding_name == seed.rdna4_encoding_name &&
         route_info.rdna4_opcode == seed.rdna4_opcode &&
         route_info.rdna4_operand_count == seed.rdna4_operand_count &&
         route_info.appears_in_rdna4_xml == seed.appears_in_rdna4_xml &&
         route_info.is_target_specific == seed.is_target_specific;
}

std::uint32_t ExpectedRoutePriority(StubDecoderRoute route) {
  switch (route) {
    case StubDecoderRoute::kVop3p:
      return 1;
    case StubDecoderRoute::kMimgTensor:
      return 2;
    case StubDecoderRoute::kVop1:
      return 3;
    case StubDecoderRoute::kVop3Sdst:
      return 4;
    case StubDecoderRoute::kUnsupported:
      break;
  }
  return 0;
}

bool RouteInstructionListMatchesSeedCatalogOrder(StubDecoderRoute route) {
  std::vector<std::string_view> expected;
  for (const DecoderSeedInfo& seed : GetDecoderSeedInfos()) {
    if (SelectStubDecoderRoute(seed.decode_hint) == route) {
      expected.push_back(seed.instruction_name);
    }
  }
  std::sort(expected.begin(), expected.end());

  const auto routed = GetStubDecoderRouteInstructions(route);
  if (expected.size() != routed.size()) {
    return false;
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != routed[i]) {
      return false;
    }
  }
  return true;
}

bool RouteInfoSequenceMatchesSeedCatalogOrder() {
  std::vector<const DecoderSeedInfo*> expected;
  for (const DecoderSeedInfo& seed : GetDecoderSeedInfos()) {
    if (SelectStubDecoderRoute(seed.decode_hint) != StubDecoderRoute::kUnsupported) {
      expected.push_back(&seed);
    }
  }
  std::sort(expected.begin(), expected.end(),
            [](const DecoderSeedInfo* lhs, const DecoderSeedInfo* rhs) {
              const StubDecoderRoute lhs_route =
                  SelectStubDecoderRoute(lhs->decode_hint);
              const StubDecoderRoute rhs_route =
                  SelectStubDecoderRoute(rhs->decode_hint);
              const std::uint32_t lhs_priority = ExpectedRoutePriority(lhs_route);
              const std::uint32_t rhs_priority = ExpectedRoutePriority(rhs_route);
              if (lhs_priority != rhs_priority) {
                return lhs_priority < rhs_priority;
              }
              return lhs->instruction_name < rhs->instruction_name;
            });

  const auto route_infos = GetStubDecoderRouteInfos();
  if (expected.size() != route_infos.size()) {
    return false;
  }
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (!MatchesSeedCatalogParity(route_infos[i], *expected[i])) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!Expect(!GetStubDecoderRouteInfos().empty(),
              "expected non-empty gfx1250 stub decoder route table")) {
    return 1;
  }
  if (!Expect(GetStubDecoderRouteManifests().size() == 4,
              "expected four prioritized stub decoder routes")) {
    return 1;
  }

  const StubDecoderRouteInfo* vop3p = FindStubDecoderRouteInfo("V_PK_ADD_BF16");
  if (!Expect(vop3p != nullptr, "expected VOP3P stub route lookup")) {
    return 1;
  }
  if (!Expect(vop3p->route == StubDecoderRoute::kVop3p,
              "expected VOP3P route for V_PK_ADD_BF16")) {
    return 1;
  }
  if (!Expect(vop3p->route_priority == 1,
              "expected VOP3P route to have highest priority")) {
    return 1;
  }

  const StubDecoderRouteInfo* tensor =
      FindStubDecoderRouteInfo("TENSOR_LOAD_TO_LDS");
  if (!Expect(tensor != nullptr, "expected tensor stub route lookup")) {
    return 1;
  }
  if (!Expect(tensor->route == StubDecoderRoute::kMimgTensor,
              "expected tensor route for TENSOR_LOAD_TO_LDS")) {
    return 1;
  }
  if (!Expect(tensor->route_priority == 2,
              "expected tensor route to be second priority")) {
    return 1;
  }

  const StubDecoderRouteInfo* vop1 = FindStubDecoderRouteInfo("V_CVT_F32_FP8");
  if (!Expect(vop1 != nullptr, "expected VOP1 stub route lookup")) {
    return 1;
  }
  if (!Expect(vop1->route == StubDecoderRoute::kVop1,
              "expected VOP1 route for V_CVT_F32_FP8")) {
    return 1;
  }
  if (!Expect(vop1->appears_in_rdna4_xml,
              "expected XML-backed VOP1 seed route")) {
    return 1;
  }

  const StubDecoderRouteInfo* sdst =
      FindStubDecoderRouteInfo("V_DIV_SCALE_F64");
  if (!Expect(sdst != nullptr, "expected VOP3 SDST stub route lookup")) {
    return 1;
  }
  if (!Expect(sdst->route == StubDecoderRoute::kVop3Sdst,
              "expected VOP3 SDST route for V_DIV_SCALE_F64")) {
    return 1;
  }
  if (!Expect(sdst->route_priority == 4,
              "expected VOP3 SDST route to be fourth priority")) {
    return 1;
  }
  if (!Expect(sdst->rdna4_encoding_name == "VOP3_SDST_ENC",
              "expected preserved RDNA4 encoding name for SDST route")) {
    return 1;
  }

  if (!Expect(SelectStubDecoderRoute("V_CVT_PK_FP8_F32") ==
                  StubDecoderRoute::kUnsupported,
              "expected unsupported route for current VOP3-only seed")) {
    return 1;
  }
  if (!Expect(FindStubDecoderRouteInfo("V_CVT_PK_FP8_F32") == nullptr,
              "expected no stub route info for unsupported VOP3-only seed")) {
    return 1;
  }

  if (!Expect(Contains(StubDecoderRoute::kVop3p, "V_WMMA_LD_SCALE_PAIRED_B32"),
              "expected paired WMMA scale load in VOP3P route")) {
    return 1;
  }
  if (!Expect(Contains(StubDecoderRoute::kVop1, "V_CVT_F16_FP8"),
              "expected FP8 conversion in VOP1 route")) {
    return 1;
  }

  std::size_t manifest_total = 0;
  for (const StubDecoderRouteManifest& manifest : GetStubDecoderRouteManifests()) {
    manifest_total += manifest.instruction_count;
  }
  if (!Expect(manifest_total == GetStubDecoderRouteInfos().size(),
              "expected manifest counts to match route info table size")) {
    return 1;
  }
  for (const StubDecoderRouteManifest& manifest : GetStubDecoderRouteManifests()) {
    if (!Expect(manifest.instruction_count ==
                        CountSeededInstructionsForRoute(manifest.route) &&
                    manifest.xml_backed_count ==
                        CountXmlBackedSeededInstructionsForRoute(manifest.route) &&
                    manifest.llvm_only_count ==
                        CountLlvmOnlySeededInstructionsForRoute(manifest.route) &&
                    manifest.target_specific_count ==
                        CountTargetSpecificSeededInstructionsForRoute(
                            manifest.route),
                "expected route manifest counts to match the routed seed-catalog slice")) {
      return 1;
    }
    if (!Expect(RouteInstructionListMatchesSeedCatalogOrder(manifest.route),
                "expected routed instruction list order to match the sorted seed-catalog slice")) {
      return 1;
    }
  }

  const StubDecoderRouteManifest* vop3p_manifest =
      FindStubDecoderRouteManifest(StubDecoderRoute::kVop3p);
  if (!Expect(vop3p_manifest != nullptr, "expected VOP3P route manifest")) {
    return 1;
  }
  if (!Expect(vop3p_manifest->route_priority == 1,
              "expected VOP3P route manifest priority")) {
    return 1;
  }
  if (!Expect(vop3p_manifest->instruction_count > 0,
              "expected populated VOP3P route manifest")) {
    return 1;
  }

  const StubDecoderRouteManifest* tensor_manifest =
      FindStubDecoderRouteManifest(StubDecoderRoute::kMimgTensor);
  if (!Expect(tensor_manifest != nullptr, "expected tensor route manifest")) {
    return 1;
  }
  if (!Expect(tensor_manifest->instruction_count == 2,
              "expected two tensor route seeds")) {
    return 1;
  }

  if (!Expect(GetStubDecoderRouteName(StubDecoderRoute::kUnsupported) ==
                  "kUnsupported",
              "expected unsupported route name")) {
    return 1;
  }

  for (const DecoderSeedInfo& seed : GetDecoderSeedInfos()) {
    const StubDecoderRoute expected_route = SelectStubDecoderRoute(seed.decode_hint);
    if (expected_route == StubDecoderRoute::kUnsupported) {
      if (!Expect(SelectStubDecoderRoute(seed.instruction_name) ==
                          StubDecoderRoute::kUnsupported &&
                      FindStubDecoderRouteInfo(seed.instruction_name) == nullptr &&
                      !ListedInAnyRoute(seed.instruction_name),
                  "expected unsupported seed to stay excluded from all routed selector surfaces")) {
        return 1;
      }
      continue;
    }

    const StubDecoderRouteInfo* route_info =
        FindStubDecoderRouteInfo(seed.instruction_name);
    if (!Expect(route_info != nullptr &&
                    SelectStubDecoderRoute(seed.instruction_name) ==
                        expected_route &&
                    Contains(expected_route, seed.instruction_name) &&
                    MatchesSeedCatalogParity(*route_info, seed),
                "expected routed seed to keep exact seed-catalog parity across selector surfaces")) {
      return 1;
    }
  }

  if (!Expect(RouteInfoSequenceMatchesSeedCatalogOrder(),
              "expected route-info sequence to match the priority-sorted routed seed-catalog slice")) {
    return 1;
  }

  return 0;
}
