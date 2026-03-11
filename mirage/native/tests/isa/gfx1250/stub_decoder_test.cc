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
using mirage::sim::isa::gfx1250::GetStubDecoderRouteInfos;
using mirage::sim::isa::gfx1250::StubDecodedInstruction;
using mirage::sim::isa::gfx1250::StubDecodeStatus;
using mirage::sim::isa::gfx1250::StubDecoderEntrypointManifest;
using mirage::sim::isa::gfx1250::StubDecoderRoute;

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
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

  const StubDecodedInstruction vop1 = DecodeVop1Stub("V_CVT_F16_FP8");
  if (!Expect(vop1.status == StubDecodeStatus::kDecodedStub,
              "expected FP8 conversion to decode through VOP1 stub")) {
    return 1;
  }
  if (!Expect(vop1.route_priority == 3,
              "expected VOP1 stub priority to be preserved")) {
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

  return 0;
}
