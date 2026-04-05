#include <algorithm>
#include <iostream>
#include <string_view>
#include <vector>

#include "lib/sim/isa/gfx1250/decoder_seed_catalog.h"
#include "lib/sim/isa/gfx1250/stub_decoder_selector.h"

namespace {

using mirage::sim::isa::gfx1250::DecodeSeedHint;
using mirage::sim::isa::gfx1250::DecoderSeedInfo;
using mirage::sim::isa::gfx1250::FindDecoderSeedInfo;
using mirage::sim::isa::gfx1250::FindSeedFamilyManifest;
using mirage::sim::isa::gfx1250::GetDecoderSeedInfos;
using mirage::sim::isa::gfx1250::FindStubDecoderRouteInfo;
using mirage::sim::isa::gfx1250::FindStubDecoderRouteManifest;
using mirage::sim::isa::gfx1250::GetSeededInstructionNames;
using mirage::sim::isa::gfx1250::GetStubDecoderRouteInfos;
using mirage::sim::isa::gfx1250::GetStubDecoderRouteInstructions;
using mirage::sim::isa::gfx1250::GetStubDecoderRouteManifests;
using mirage::sim::isa::gfx1250::GetStubDecoderRouteName;
using mirage::sim::isa::gfx1250::SelectStubDecoderRoute;
using mirage::sim::isa::gfx1250::StubDecoderRoute;
using mirage::sim::isa::gfx1250::StubDecoderRouteInfo;
using mirage::sim::isa::gfx1250::StubDecoderRouteManifest;
using mirage::sim::isa::gfx1250::SeedFamily;
using mirage::sim::isa::gfx1250::SeedFamilyManifest;

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

StubDecoderRoute ExpectedRouteForDecodeHint(DecodeSeedHint decode_hint) {
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

std::string_view ExpectedRouteName(StubDecoderRoute route) {
  switch (route) {
    case StubDecoderRoute::kVop3p:
      return "kVop3p";
    case StubDecoderRoute::kMimgTensor:
      return "kMimgTensor";
    case StubDecoderRoute::kVop1:
      return "kVop1";
    case StubDecoderRoute::kVop3Sdst:
      return "kVop3Sdst";
    case StubDecoderRoute::kUnsupported:
      break;
  }
  return "kUnsupported";
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
    if (ExpectedRouteForDecodeHint(seed.decode_hint) == route) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountXmlBackedSeededInstructionsForRoute(StubDecoderRoute route) {
  std::uint32_t count = 0;
  for (const DecoderSeedInfo& seed : GetDecoderSeedInfos()) {
    if (ExpectedRouteForDecodeHint(seed.decode_hint) == route &&
        seed.appears_in_rdna4_xml) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountLlvmOnlySeededInstructionsForRoute(StubDecoderRoute route) {
  std::uint32_t count = 0;
  for (const DecoderSeedInfo& seed : GetDecoderSeedInfos()) {
    if (ExpectedRouteForDecodeHint(seed.decode_hint) == route &&
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
    if (ExpectedRouteForDecodeHint(seed.decode_hint) == route &&
        seed.is_target_specific) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountTotalXmlBackedSeededInstructions() {
  std::uint32_t count = 0;
  for (const DecoderSeedInfo& seed : GetDecoderSeedInfos()) {
    if (seed.appears_in_rdna4_xml) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountTotalLlvmOnlySeededInstructions() {
  std::uint32_t count = 0;
  for (const DecoderSeedInfo& seed : GetDecoderSeedInfos()) {
    if (!seed.appears_in_rdna4_xml) {
      ++count;
    }
  }
  return count;
}

std::uint32_t CountTotalTargetSpecificSeededInstructions() {
  std::uint32_t count = 0;
  for (const DecoderSeedInfo& seed : GetDecoderSeedInfos()) {
    if (seed.is_target_specific) {
      ++count;
    }
  }
  return count;
}

bool MatchesSeedCatalogParity(const StubDecoderRouteInfo& route_info,
                              const DecoderSeedInfo& seed) {
  const StubDecoderRoute expected_route =
      ExpectedRouteForDecodeHint(seed.decode_hint);
  const StubDecoderRouteManifest* manifest =
      FindStubDecoderRouteManifest(expected_route);
  if (manifest == nullptr) {
    return false;
  }
  return route_info.instruction_name == seed.instruction_name &&
         route_info.route == expected_route &&
         route_info.route_name == ExpectedRouteName(expected_route) &&
         route_info.route_name == manifest->route_name &&
         route_info.route_priority == ExpectedRoutePriority(expected_route) &&
         route_info.route_priority == manifest->route_priority &&
         route_info.decode_hint == seed.decode_hint &&
         route_info.rdna4_encoding_name == seed.rdna4_encoding_name &&
         route_info.rdna4_opcode == seed.rdna4_opcode &&
         route_info.rdna4_operand_count == seed.rdna4_operand_count &&
         route_info.appears_in_rdna4_xml == seed.appears_in_rdna4_xml &&
         route_info.is_target_specific == seed.is_target_specific;
}

bool RouteInstructionListMatchesSeedCatalogOrder(StubDecoderRoute route) {
  std::vector<std::string_view> expected;
  for (const DecoderSeedInfo& seed : GetDecoderSeedInfos()) {
    if (ExpectedRouteForDecodeHint(seed.decode_hint) == route) {
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
    if (ExpectedRouteForDecodeHint(seed.decode_hint) !=
        StubDecoderRoute::kUnsupported) {
      expected.push_back(&seed);
    }
  }
  std::sort(expected.begin(), expected.end(),
            [](const DecoderSeedInfo* lhs, const DecoderSeedInfo* rhs) {
              const StubDecoderRoute lhs_route =
                  ExpectedRouteForDecodeHint(lhs->decode_hint);
              const StubDecoderRoute rhs_route =
                  ExpectedRouteForDecodeHint(rhs->decode_hint);
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

bool RouteManifestAccountingAndMetadataAreInternallyConsistent() {
  const auto manifests = GetStubDecoderRouteManifests();
  for (std::size_t i = 0; i < manifests.size(); ++i) {
    if (manifests[i].instruction_count == 0 ||
        manifests[i].instruction_count !=
            manifests[i].xml_backed_count + manifests[i].llvm_only_count ||
        manifests[i].target_specific_count > manifests[i].instruction_count) {
      return false;
    }
    for (std::size_t j = i + 1; j < manifests.size(); ++j) {
      if (manifests[i].route_name == manifests[j].route_name ||
          manifests[i].route_priority == manifests[j].route_priority) {
        return false;
      }
    }
  }
  return true;
}

bool RouteManifestSequenceMatchesSupportedRouteOrder() {
  static constexpr struct {
    StubDecoderRoute route;
    std::string_view route_name;
    std::uint32_t route_priority;
  } kExpectedRoutes[] = {
      {StubDecoderRoute::kVop3p, "kVop3p", 1},
      {StubDecoderRoute::kMimgTensor, "kMimgTensor", 2},
      {StubDecoderRoute::kVop1, "kVop1", 3},
      {StubDecoderRoute::kVop3Sdst, "kVop3Sdst", 4},
  };

  const auto manifests = GetStubDecoderRouteManifests();
  if (manifests.size() !=
      sizeof(kExpectedRoutes) / sizeof(kExpectedRoutes[0])) {
    return false;
  }

  for (std::size_t i = 0; i < manifests.size(); ++i) {
    if (manifests[i].route != kExpectedRoutes[i].route ||
        manifests[i].route_name != kExpectedRoutes[i].route_name ||
        manifests[i].route_priority != kExpectedRoutes[i].route_priority ||
        FindStubDecoderRouteManifest(manifests[i].route) != &manifests[i]) {
      return false;
    }
  }
  return true;
}

bool RouteManifestBoundariesMatchSelectorSurfaces() {
  const auto manifests = GetStubDecoderRouteManifests();
  const auto route_infos = GetStubDecoderRouteInfos();

  std::size_t route_info_offset = 0;
  for (const StubDecoderRouteManifest& manifest : manifests) {
    const auto routed_instructions =
        GetStubDecoderRouteInstructions(manifest.route);
    if (manifest.instruction_count == 0 ||
        routed_instructions.size() != manifest.instruction_count ||
        route_info_offset + manifest.instruction_count > route_infos.size()) {
      return false;
    }

    if (FindStubDecoderRouteInfo(routed_instructions.front()) !=
            &route_infos[route_info_offset] ||
        FindStubDecoderRouteInfo(routed_instructions.back()) !=
            &route_infos[route_info_offset + manifest.instruction_count - 1]) {
      return false;
    }

    for (std::size_t i = 0; i < routed_instructions.size(); ++i) {
      const StubDecoderRouteInfo& route_info = route_infos[route_info_offset + i];
      if (route_info.route != manifest.route ||
          route_info.route_name != manifest.route_name ||
          route_info.route_priority != manifest.route_priority ||
          route_info.instruction_name != routed_instructions[i]) {
        return false;
      }
    }

    route_info_offset += manifest.instruction_count;
  }

  return route_info_offset == route_infos.size();
}

bool UnsupportedSeededSliceMatchesExcludedSelectorSurface() {
  std::uint32_t excluded_count = 0;
  for (const DecoderSeedInfo& seed : GetDecoderSeedInfos()) {
    if (ExpectedRouteForDecodeHint(seed.decode_hint) !=
        StubDecoderRoute::kUnsupported) {
      continue;
    }
    if (SelectStubDecoderRoute(seed.instruction_name) !=
            StubDecoderRoute::kUnsupported ||
        FindStubDecoderRouteInfo(seed.instruction_name) != nullptr ||
        ListedInAnyRoute(seed.instruction_name)) {
      return false;
    }
    ++excluded_count;
  }
  return excluded_count ==
         CountSeededInstructionsForRoute(StubDecoderRoute::kUnsupported);
}

bool ScalePairedFamilyManifestMatchesSeedCatalog() {
  const SeedFamilyManifest* manifest =
      FindSeedFamilyManifest(SeedFamily::kScalePaired);
  if (manifest == nullptr) {
    return false;
  }

  const auto seeded_instructions =
      GetSeededInstructionNames(SeedFamily::kScalePaired);
  return seeded_instructions.size() == 52 &&
         manifest->seeded_instruction_count == 52 &&
         manifest->xml_backed_count == 1 &&
         manifest->llvm_only_count == 51 &&
         manifest->target_specific_count == 51 &&
         manifest->vop1_hint_count == 0 &&
         manifest->vop3_hint_count == 45 &&
         manifest->vop3p_hint_count == 6 &&
         manifest->vop3_sdst_hint_count == 1 &&
         manifest->mimg_tensor_hint_count == 0 &&
         seeded_instructions.front() == "V_WMMA_LD_SCALE_PAIRED_B32" &&
         seeded_instructions[1] == "V_WMMA_LD_SCALE16_PAIRED_B64" &&
         seeded_instructions[2] == "V_CVT_SCALEF32_PK16_BF6_BF16" &&
         seeded_instructions.back() == "V_WMMA_SCALE_F32_32X16X128_F4_w32";
}

bool ScalePairedFamilyRouteSurfaceMatchesSeedCatalog() {
  const auto seeded_instructions =
      GetSeededInstructionNames(SeedFamily::kScalePaired);
  for (const std::string_view instruction_name : seeded_instructions) {
    const DecoderSeedInfo* seed = FindDecoderSeedInfo(instruction_name);
    if (seed == nullptr) {
      return false;
    }

    const StubDecoderRoute expected_route =
        ExpectedRouteForDecodeHint(seed->decode_hint);
    const StubDecoderRouteInfo* route_info =
        FindStubDecoderRouteInfo(instruction_name);
    if (expected_route == StubDecoderRoute::kUnsupported) {
      if (SelectStubDecoderRoute(instruction_name) !=
              StubDecoderRoute::kUnsupported ||
          route_info != nullptr || ListedInAnyRoute(instruction_name)) {
        return false;
      }
      continue;
    }

    if (route_info == nullptr || route_info->route != expected_route ||
        SelectStubDecoderRoute(instruction_name) != expected_route ||
        !Contains(expected_route, instruction_name) ||
        !MatchesSeedCatalogParity(*route_info, *seed)) {
      return false;
    }
  }
  return true;
}

bool ScalePairedTailBatchMatchesSeedCatalog() {
  const auto seeded_instructions =
      GetSeededInstructionNames(SeedFamily::kScalePaired);
  if (seeded_instructions.size() != 52 ||
      seeded_instructions[2] != "V_CVT_SCALEF32_PK16_BF6_BF16" ||
      seeded_instructions.back() != "V_WMMA_SCALE_F32_32X16X128_F4_w32") {
    return false;
  }

  for (std::size_t i = 2; i < seeded_instructions.size(); ++i) {
    const std::string_view instruction_name = seeded_instructions[i];
    const DecoderSeedInfo* seed = FindDecoderSeedInfo(instruction_name);
    if (seed == nullptr) {
      return false;
    }

    const StubDecoderRoute expected_route =
        ExpectedRouteForDecodeHint(seed->decode_hint);
    const StubDecoderRouteInfo* route_info =
        FindStubDecoderRouteInfo(instruction_name);
    if (expected_route == StubDecoderRoute::kUnsupported) {
      if (SelectStubDecoderRoute(instruction_name) !=
              StubDecoderRoute::kUnsupported ||
          route_info != nullptr || ListedInAnyRoute(instruction_name)) {
        return false;
      }
      continue;
    }

    if (route_info == nullptr || route_info->route != expected_route ||
        SelectStubDecoderRoute(instruction_name) != expected_route ||
        !Contains(expected_route, instruction_name) ||
        !MatchesSeedCatalogParity(*route_info, *seed)) {
      return false;
    }
  }
  return true;
}

bool Fp8Bf8LeadingBatchRouteSurfaceMatchesSeedCatalog() {
  const auto seeded_instructions =
      GetSeededInstructionNames(SeedFamily::kFp8Bf8);
  if (seeded_instructions.size() != 87 ||
      seeded_instructions.front() != "V_CVT_F16_FP8" ||
      seeded_instructions[49] != "V_CVT_SCALE_PK8_F16_FP8" ||
      seeded_instructions[50] != "V_CVT_SCALE_PK8_F32_BF8" ||
      seeded_instructions.back() != "V_WMMA_SCALE_F32_32X16X128_F4_w32") {
    return false;
  }

  for (std::size_t i = 0; i < 50; ++i) {
    const std::string_view instruction_name = seeded_instructions[i];
    const DecoderSeedInfo* seed = FindDecoderSeedInfo(instruction_name);
    if (seed == nullptr) {
      return false;
    }

    const StubDecoderRoute expected_route =
        ExpectedRouteForDecodeHint(seed->decode_hint);
    const StubDecoderRouteInfo* route_info =
        FindStubDecoderRouteInfo(instruction_name);
    if (expected_route == StubDecoderRoute::kUnsupported) {
      if (SelectStubDecoderRoute(instruction_name) !=
              StubDecoderRoute::kUnsupported ||
          route_info != nullptr || ListedInAnyRoute(instruction_name)) {
        return false;
      }
      continue;
    }

    if (route_info == nullptr || route_info->route != expected_route ||
        SelectStubDecoderRoute(instruction_name) != expected_route ||
        !Contains(expected_route, instruction_name) ||
        !MatchesSeedCatalogParity(*route_info, *seed)) {
      return false;
    }
  }
  return true;
}

bool Fp8Bf8TailBatchRouteSurfaceMatchesSeedCatalog() {
  const auto seeded_instructions =
      GetSeededInstructionNames(SeedFamily::kFp8Bf8);
  if (seeded_instructions.size() != 87 ||
      seeded_instructions[50] != "V_CVT_SCALE_PK8_F32_BF8" ||
      seeded_instructions.back() != "V_WMMA_SCALE_F32_32X16X128_F4_w32") {
    return false;
  }

  for (std::size_t i = 50; i < seeded_instructions.size(); ++i) {
    const std::string_view instruction_name = seeded_instructions[i];
    const DecoderSeedInfo* seed = FindDecoderSeedInfo(instruction_name);
    if (seed == nullptr) {
      return false;
    }

    const StubDecoderRoute expected_route =
        ExpectedRouteForDecodeHint(seed->decode_hint);
    const StubDecoderRouteInfo* route_info =
        FindStubDecoderRouteInfo(instruction_name);
    if (expected_route == StubDecoderRoute::kUnsupported) {
      if (SelectStubDecoderRoute(instruction_name) !=
              StubDecoderRoute::kUnsupported ||
          route_info != nullptr || ListedInAnyRoute(instruction_name)) {
        return false;
      }
      continue;
    }

    if (route_info == nullptr || route_info->route != expected_route ||
        SelectStubDecoderRoute(instruction_name) != expected_route ||
        !Contains(expected_route, instruction_name) ||
        !MatchesSeedCatalogParity(*route_info, *seed)) {
      return false;
    }
  }
  return true;
}

bool Vop3pLeadingBatchRouteSurfaceMatchesSeedCatalog() {
  const auto seeded_instructions =
      GetSeededInstructionNames(SeedFamily::kVop3p);
  if (seeded_instructions.size() != 62 ||
      seeded_instructions.front() != "V_PK_ADD_BF16" ||
      seeded_instructions[49] != "V_WMMA_F32_16X16X4_F32_w32" ||
      seeded_instructions[50] != "V_WMMA_F32_16X16X64_BF8_BF8_w32" ||
      seeded_instructions.back() != "V_WMMA_SCALE_F32_32X16X128_F4_w32") {
    return false;
  }

  for (std::size_t i = 0; i < 50; ++i) {
    const std::string_view instruction_name = seeded_instructions[i];
    const DecoderSeedInfo* seed = FindDecoderSeedInfo(instruction_name);
    if (seed == nullptr) {
      return false;
    }

    const StubDecoderRoute expected_route =
        ExpectedRouteForDecodeHint(seed->decode_hint);
    const StubDecoderRouteInfo* route_info =
        FindStubDecoderRouteInfo(instruction_name);
    if (expected_route != StubDecoderRoute::kVop3p ||
        route_info == nullptr || route_info->route != expected_route ||
        SelectStubDecoderRoute(instruction_name) != expected_route ||
        !Contains(expected_route, instruction_name) ||
        !MatchesSeedCatalogParity(*route_info, *seed)) {
      return false;
    }
  }
  return true;
}

bool Fp8Bf8FamilyManifestMatchesSeedCatalog() {
  const SeedFamilyManifest* manifest =
      FindSeedFamilyManifest(SeedFamily::kFp8Bf8);
  if (manifest == nullptr) {
    return false;
  }

  const auto seeded_instructions =
      GetSeededInstructionNames(SeedFamily::kFp8Bf8);
  return seeded_instructions.size() == 87 &&
         manifest->seeded_instruction_count == 87 &&
         manifest->xml_backed_count == 3 &&
         manifest->llvm_only_count == 84 &&
         manifest->target_specific_count == 84 &&
         manifest->vop1_hint_count == 5 &&
         manifest->vop3_hint_count == 52 &&
         manifest->vop3p_hint_count == 30 &&
         manifest->vop3_sdst_hint_count == 0 &&
         manifest->mimg_tensor_hint_count == 0 &&
         seeded_instructions.front() == "V_CVT_F16_FP8" &&
         seeded_instructions[49] == "V_CVT_SCALE_PK8_F16_FP8" &&
         seeded_instructions[50] == "V_CVT_SCALE_PK8_F32_BF8" &&
         seeded_instructions.back() == "V_WMMA_SCALE_F32_32X16X128_F4_w32";
}

bool Vop3pFamilyManifestMatchesSeedCatalog() {
  const SeedFamilyManifest* manifest =
      FindSeedFamilyManifest(SeedFamily::kVop3p);
  if (manifest == nullptr) {
    return false;
  }

  const auto seeded_instructions =
      GetSeededInstructionNames(SeedFamily::kVop3p);
  return seeded_instructions.size() == 62 &&
         manifest->seeded_instruction_count == 62 &&
         manifest->xml_backed_count == 0 &&
         manifest->llvm_only_count == 62 &&
         manifest->target_specific_count == 62 &&
         manifest->vop1_hint_count == 0 &&
         manifest->vop3_hint_count == 0 &&
         manifest->vop3p_hint_count == 62 &&
         manifest->vop3_sdst_hint_count == 0 &&
         manifest->mimg_tensor_hint_count == 0 &&
         seeded_instructions.front() == "V_PK_ADD_BF16" &&
         seeded_instructions[49] == "V_WMMA_F32_16X16X4_F32_w32" &&
         seeded_instructions[50] == "V_WMMA_F32_16X16X64_BF8_BF8_w32" &&
         seeded_instructions.back() == "V_WMMA_SCALE_F32_32X16X128_F4_w32";
}

bool Vop3pTailBatchMatchesSeedCatalog() {
  const SeedFamilyManifest* manifest =
      FindSeedFamilyManifest(SeedFamily::kVop3p);
  if (manifest == nullptr) {
    return false;
  }

  const auto seeded_instructions =
      GetSeededInstructionNames(SeedFamily::kVop3p);
  if (seeded_instructions.size() != 62 ||
      manifest->seeded_instruction_count != 62 ||
      manifest->vop3p_hint_count != 62 ||
      seeded_instructions[50] != "V_WMMA_F32_16X16X64_BF8_BF8_w32" ||
      seeded_instructions[51] != "V_WMMA_F32_16X16X64_BF8_FP8_w32" ||
      seeded_instructions[52] != "V_WMMA_F32_16X16X64_FP8_BF8_w32" ||
      seeded_instructions[53] != "V_WMMA_F32_16X16X64_FP8_FP8_w32" ||
      seeded_instructions[54] != "V_WMMA_F32_32X16X128_F4_w32" ||
      seeded_instructions[55] != "V_WMMA_I32_16X16X64_IU8_w32" ||
      seeded_instructions[56] != "V_WMMA_LD_SCALE16_PAIRED_B64" ||
      seeded_instructions[57] != "V_WMMA_LD_SCALE_PAIRED_B32" ||
      seeded_instructions[58] != "V_WMMA_SCALE16_F32_16X16X128_F8F6F4" ||
      seeded_instructions[59] != "V_WMMA_SCALE16_F32_32X16X128_F4_w32" ||
      seeded_instructions[60] != "V_WMMA_SCALE_F32_16X16X128_F8F6F4" ||
      seeded_instructions[61] != "V_WMMA_SCALE_F32_32X16X128_F4_w32") {
    return false;
  }

  for (std::size_t i = 50; i < seeded_instructions.size(); ++i) {
    const std::string_view instruction_name = seeded_instructions[i];
    const DecoderSeedInfo* seed = FindDecoderSeedInfo(instruction_name);
    if (seed == nullptr) {
      return false;
    }

    const StubDecoderRoute expected_route =
        ExpectedRouteForDecodeHint(seed->decode_hint);
    const StubDecoderRouteInfo* route_info =
        FindStubDecoderRouteInfo(instruction_name);
    if (expected_route != StubDecoderRoute::kVop3p || route_info == nullptr ||
        route_info->route != expected_route ||
        SelectStubDecoderRoute(instruction_name) != expected_route ||
        !Contains(expected_route, instruction_name) ||
        !MatchesSeedCatalogParity(*route_info, *seed)) {
      return false;
    }
  }
  return true;
}

bool WmmaFamilyManifestMatchesSeedCatalog() {
  const SeedFamilyManifest* manifest = FindSeedFamilyManifest(SeedFamily::kWmma);
  if (manifest == nullptr) {
    return false;
  }

  const auto seeded_instructions = GetSeededInstructionNames(SeedFamily::kWmma);
  return seeded_instructions.size() == 47 &&
         manifest->seeded_instruction_count == 47 &&
         manifest->xml_backed_count == 0 &&
         manifest->llvm_only_count == 47 &&
         manifest->target_specific_count == 47 &&
         manifest->vop1_hint_count == 0 &&
         manifest->vop3_hint_count == 0 &&
         manifest->vop3p_hint_count == 45 &&
         manifest->vop3_sdst_hint_count == 0 &&
         manifest->mimg_tensor_hint_count == 2 &&
         seeded_instructions.front() == "V_WMMA_F32_16X16X4_F32_w32" &&
         seeded_instructions[1] == "V_WMMA_BF16F32_16X16X32_BF16_w32" &&
         seeded_instructions[2] == "V_SWMMAC_F32_16X16X64_F16_w32" &&
         seeded_instructions[3] == "TENSOR_LOAD_TO_LDS" &&
         seeded_instructions[4] == "TENSOR_STORE_FROM_LDS" &&
         seeded_instructions.back() == "V_WMMA_SCALE_F32_32X16X128_F4_w32";
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
  if (!Expect(RouteManifestSequenceMatchesSupportedRouteOrder(),
              "expected route manifests to stay in exact supported-route priority order with sequence-stable lookup parity")) {
    return 1;
  }
  if (!Expect(RouteManifestBoundariesMatchSelectorSurfaces(),
              "expected route manifests to partition routed selector surfaces into exact contiguous per-route blocks")) {
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
  std::size_t manifest_xml_backed_total = 0;
  std::size_t manifest_llvm_only_total = 0;
  std::size_t manifest_target_specific_total = 0;
  for (const StubDecoderRouteManifest& manifest : GetStubDecoderRouteManifests()) {
    manifest_total += manifest.instruction_count;
    manifest_xml_backed_total += manifest.xml_backed_count;
    manifest_llvm_only_total += manifest.llvm_only_count;
    manifest_target_specific_total += manifest.target_specific_count;
  }
  if (!Expect(manifest_total == GetStubDecoderRouteInfos().size(),
              "expected manifest counts to match route info table size")) {
    return 1;
  }
  if (!Expect(
          manifest_total +
                  CountSeededInstructionsForRoute(
                      StubDecoderRoute::kUnsupported) ==
              GetDecoderSeedInfos().size(),
          "expected routed manifest totals plus unsupported seeded remainder to partition the seed catalog exactly")) {
    return 1;
  }
  if (!Expect(
          manifest_xml_backed_total +
                      CountXmlBackedSeededInstructionsForRoute(
                          StubDecoderRoute::kUnsupported) ==
                  CountTotalXmlBackedSeededInstructions() &&
              manifest_llvm_only_total +
                      CountLlvmOnlySeededInstructionsForRoute(
                          StubDecoderRoute::kUnsupported) ==
                  CountTotalLlvmOnlySeededInstructions() &&
              manifest_target_specific_total +
                      CountTargetSpecificSeededInstructionsForRoute(
                          StubDecoderRoute::kUnsupported) ==
                  CountTotalTargetSpecificSeededInstructions(),
          "expected routed manifest provenance totals plus unsupported seeded remainder to partition seed-catalog provenance counts exactly")) {
    return 1;
  }
  if (!Expect(
          RouteManifestAccountingAndMetadataAreInternallyConsistent(),
          "expected route manifests to keep exact internal count accounting and one-to-one metadata")) {
    return 1;
  }
  if (!Expect(UnsupportedSeededSliceMatchesExcludedSelectorSurface(),
              "expected unsupported seeded remainder to stay fully excluded from routed selector surfaces")) {
    return 1;
  }
  if (!Expect(ScalePairedFamilyManifestMatchesSeedCatalog(),
              "expected scale-paired family manifest to keep exact seed-catalog parity across the post-pair-load 50-slice batch and paired-load anchors")) {
    return 1;
  }
  if (!Expect(ScalePairedFamilyRouteSurfaceMatchesSeedCatalog(),
              "expected scale-paired family to keep exact route-keyed parity, selector exclusion, and manifest consistency across the post-pair-load 50-slice batch")) {
    return 1;
  }
  if (!Expect(ScalePairedTailBatchMatchesSeedCatalog(),
              "expected scale-paired family to keep exact route-keyed parity and selector exclusion across the routed 50-seed tail batch")) {
    return 1;
  }
  if (!Expect(Fp8Bf8LeadingBatchRouteSurfaceMatchesSeedCatalog(),
              "expected fp8/bf8 family to keep exact route-keyed parity and selector exclusion across the leading 50-seed batch")) {
    return 1;
  }
  if (!Expect(Fp8Bf8TailBatchRouteSurfaceMatchesSeedCatalog(),
              "expected fp8/bf8 family to keep exact route-keyed parity and selector exclusion across the remaining tail batch")) {
    return 1;
  }
  if (!Expect(Vop3pLeadingBatchRouteSurfaceMatchesSeedCatalog(),
              "expected VOP3P family to keep exact route-keyed parity and selector/manifest consistency across the leading 50-seed batch")) {
    return 1;
  }
  if (!Expect(Fp8Bf8FamilyManifestMatchesSeedCatalog(),
              "expected fp8/bf8 family manifest to keep exact seed-catalog parity across the first 50-slice batch")) {
    return 1;
  }
  if (!Expect(Vop3pFamilyManifestMatchesSeedCatalog(),
              "expected VOP3P family manifest to keep exact seed-catalog parity across the next 50-slice batch")) {
    return 1;
  }
  if (!Expect(Vop3pTailBatchMatchesSeedCatalog(),
              "expected VOP3P family manifest to keep exact seed-catalog parity across the remaining tail batch")) {
    return 1;
  }
  if (!Expect(WmmaFamilyManifestMatchesSeedCatalog(),
              "expected WMMA family manifest to keep exact seed-catalog parity across the routed tensor and WMMA batch")) {
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
                            manifest.route) &&
                    manifest.route_name == ExpectedRouteName(manifest.route) &&
                    manifest.route_name ==
                        GetStubDecoderRouteName(manifest.route) &&
                    manifest.route_priority ==
                        ExpectedRoutePriority(manifest.route),
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
  if (!Expect(GetStubDecoderRouteInstructions(StubDecoderRoute::kUnsupported)
                      .empty() &&
                  FindStubDecoderRouteManifest(StubDecoderRoute::kUnsupported) ==
                      nullptr,
              "expected unsupported route to expose no routed instruction list or manifest")) {
    return 1;
  }
  if (!Expect(SelectStubDecoderRoute(
                      static_cast<DecodeSeedHint>(99)) ==
                  StubDecoderRoute::kUnsupported,
              "expected invalid decode hint to fall back to unsupported route")) {
    return 1;
  }
  if (!Expect(GetStubDecoderRouteName(static_cast<StubDecoderRoute>(99)) ==
                      "kUnsupported" &&
                  GetStubDecoderRouteInstructions(
                      static_cast<StubDecoderRoute>(99))
                      .empty() &&
                  FindStubDecoderRouteManifest(
                      static_cast<StubDecoderRoute>(99)) == nullptr,
              "expected invalid route enum to expose unsupported selector fallbacks")) {
    return 1;
  }
  if (!Expect(SelectStubDecoderRoute("NO_SUCH_GFX1250_OPCODE") ==
                      StubDecoderRoute::kUnsupported &&
                  FindStubDecoderRouteInfo("NO_SUCH_GFX1250_OPCODE") ==
                      nullptr &&
                  !ListedInAnyRoute("NO_SUCH_GFX1250_OPCODE"),
              "expected unknown instruction to stay excluded from all routed selector surfaces")) {
    return 1;
  }
  if (!Expect(SelectStubDecoderRoute("") == StubDecoderRoute::kUnsupported &&
                  FindStubDecoderRouteInfo("") == nullptr &&
                  !ListedInAnyRoute(""),
              "expected empty instruction name to stay excluded from all routed selector surfaces")) {
    return 1;
  }
  if (!Expect(SelectStubDecoderRoute("v_pk_add_bf16") ==
                      StubDecoderRoute::kUnsupported &&
                  FindStubDecoderRouteInfo("v_pk_add_bf16") == nullptr &&
                  !ListedInAnyRoute("v_pk_add_bf16"),
              "expected lowercase known opcode to stay excluded from all routed selector surfaces")) {
    return 1;
  }
  for (std::string_view padded_instruction :
       {" V_PK_ADD_BF16", "V_PK_ADD_BF16 "}) {
    if (!Expect(SelectStubDecoderRoute(padded_instruction) ==
                        StubDecoderRoute::kUnsupported &&
                    FindStubDecoderRouteInfo(padded_instruction) == nullptr &&
                    !ListedInAnyRoute(padded_instruction),
                "expected whitespace-padded known opcode to stay excluded from all routed selector surfaces")) {
      return 1;
    }
  }
  for (std::string_view near_miss_instruction :
       {"X_V_PK_ADD_BF16",
        "X_TENSOR_LOAD_TO_LDS",
        "X_V_CVT_F16_FP8",
        "X_V_DIV_SCALE_F64"}) {
    if (!Expect(SelectStubDecoderRoute(near_miss_instruction) ==
                        StubDecoderRoute::kUnsupported &&
                    FindStubDecoderRouteInfo(near_miss_instruction) ==
                        nullptr &&
                    !ListedInAnyRoute(near_miss_instruction),
                "expected prefixed known opcode near-misses to stay excluded from all routed selector surfaces")) {
      return 1;
    }
  }
  for (std::string_view near_miss_instruction :
       {"V_PK_ADD_BF16_X",
        "TENSOR_LOAD_TO_LDS_X",
        "V_CVT_F16_FP8_X",
        "V_DIV_SCALE_F64_X",
        "X_V_PK_ADD_BF16_X",
        "X_TENSOR_LOAD_TO_LDS_X",
        "X_V_CVT_F16_FP8_X",
        "X_V_DIV_SCALE_F64_X"}) {
    if (!Expect(SelectStubDecoderRoute(near_miss_instruction) ==
                        StubDecoderRoute::kUnsupported &&
                    FindStubDecoderRouteInfo(near_miss_instruction) ==
                        nullptr &&
                    !ListedInAnyRoute(near_miss_instruction),
                "expected suffixed known opcode near-misses to stay excluded from all routed selector surfaces")) {
      return 1;
    }
  }
  for (std::string_view decorated_near_miss_instruction :
       {"x_v_pk_add_bf16",
        "x_tensor_load_to_lds",
        "x_v_cvt_f16_fp8",
        "x_v_div_scale_f64",
        "v_pk_add_bf16_x",
        "tensor_load_to_lds_x",
        "v_cvt_f16_fp8_x",
        "v_div_scale_f64_x",
        "x_v_pk_add_bf16_x",
        "x_tensor_load_to_lds_x",
        "x_v_cvt_f16_fp8_x",
        "x_v_div_scale_f64_x",
        " X_V_PK_ADD_BF16",
        "X_V_PK_ADD_BF16 ",
        " X_TENSOR_LOAD_TO_LDS",
        "TENSOR_LOAD_TO_LDS_X ",
        " X_V_CVT_F16_FP8_X",
        "X_V_DIV_SCALE_F64_X "}) {
    if (!Expect(SelectStubDecoderRoute(decorated_near_miss_instruction) ==
                        StubDecoderRoute::kUnsupported &&
                    FindStubDecoderRouteInfo(decorated_near_miss_instruction) ==
                        nullptr &&
                    !ListedInAnyRoute(decorated_near_miss_instruction),
                "expected decorated known opcode near-misses to stay excluded from all routed selector surfaces")) {
      return 1;
    }
  }
  for (std::string_view split_token_near_miss_instruction :
       {"V_PK__ADD_BF16",
        "V_CVT_F16__FP8",
        "V_CVT_PK__F16_FP8",
        "V_WMMA__F32_16X16X4_F32_w32",
        "V_WMMA_SCALE__F32_16X16X128_F8F6F4"}) {
    if (!Expect(
            SelectStubDecoderRoute(split_token_near_miss_instruction) ==
                        StubDecoderRoute::kUnsupported &&
                    FindStubDecoderRouteInfo(split_token_near_miss_instruction) ==
                        nullptr &&
                    !ListedInAnyRoute(split_token_near_miss_instruction),
            "expected split-token family near-misses to stay excluded from all routed selector surfaces")) {
      return 1;
    }
  }
  for (std::string_view delayed_split_token_near_miss_instruction :
       {"V_PK_ADD__BF16",
        "V_CVT__F16_FP8",
        "V_CVT_PK_F16__FP8",
        "V_WMMA_F32__16X16X4_F32_w32",
        "V_WMMA_SCALE_F32__16X16X128_F8F6F4"}) {
    if (!Expect(
            SelectStubDecoderRoute(delayed_split_token_near_miss_instruction) ==
                        StubDecoderRoute::kUnsupported &&
                    FindStubDecoderRouteInfo(
                        delayed_split_token_near_miss_instruction) ==
                        nullptr &&
                    !ListedInAnyRoute(
                        delayed_split_token_near_miss_instruction),
            "expected delayed split-token family near-misses to stay excluded from all routed selector surfaces")) {
      return 1;
    }
  }

  for (const DecoderSeedInfo& seed : GetDecoderSeedInfos()) {
    const StubDecoderRoute expected_route =
        ExpectedRouteForDecodeHint(seed.decode_hint);
    if (expected_route == StubDecoderRoute::kUnsupported) {
      if (!Expect(SelectStubDecoderRoute(seed.decode_hint) ==
                          StubDecoderRoute::kUnsupported &&
                      SelectStubDecoderRoute(seed.instruction_name) ==
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
                    SelectStubDecoderRoute(seed.decode_hint) ==
                        expected_route &&
                    SelectStubDecoderRoute(seed.instruction_name) ==
                        expected_route &&
                    GetStubDecoderRouteName(expected_route) ==
                        ExpectedRouteName(expected_route) &&
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
