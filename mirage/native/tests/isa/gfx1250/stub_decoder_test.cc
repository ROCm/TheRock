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
using mirage::sim::isa::gfx1250::GetStubOpcodeShapeName;
using mirage::sim::isa::gfx1250::GetStubDecoderRouteInfos;
using mirage::sim::isa::gfx1250::StubDecodedInstruction;
using mirage::sim::isa::gfx1250::StubDecodeStatus;
using mirage::sim::isa::gfx1250::StubDecoderEntrypointManifest;
using mirage::sim::isa::gfx1250::StubDecoderRoute;
using mirage::sim::isa::gfx1250::StubExecutionDomain;
using mirage::sim::isa::gfx1250::StubOperandLayoutKind;
using mirage::sim::isa::gfx1250::StubOpcodeShape;

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
                  StubOperandLayoutKind::kUnknown,
              "expected no explicit operand layout for V_CVT_F16_FP8 yet")) {
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
                  StubOperandLayoutKind::kUnknown,
              "expected no explicit operand layout for V_DIV_SCALE_F64 yet")) {
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

  return 0;
}
