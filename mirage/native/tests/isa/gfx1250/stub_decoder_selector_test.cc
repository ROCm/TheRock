#include <iostream>
#include <string_view>

#include "lib/sim/isa/gfx1250/stub_decoder_selector.h"

namespace {

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

  return 0;
}
