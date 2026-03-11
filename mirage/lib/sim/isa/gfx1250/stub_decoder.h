#ifndef MIRAGE_SIM_ISA_GFX1250_STUB_DECODER_H_
#define MIRAGE_SIM_ISA_GFX1250_STUB_DECODER_H_

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

#include "lib/sim/isa/gfx1250/stub_decoder_selector.h"

namespace mirage::sim::isa::gfx1250 {

enum class StubDecodeStatus {
  kDecodedStub,
  kUnsupportedRoute,
  kUnknownInstruction,
};

enum class StubOpcodeShape {
  kUnknown,
  kVop3pPackedBinary,
  kVop3pPackedFma,
  kWmmaCore,
  kWmmaScale,
  kWmmaScalePairedLoad,
  kSwmmacCore,
  kTensorLoadToLds,
  kTensorStoreFromLds,
  kFp8ConvertToF16,
  kFp8ConvertToF32,
  kFp8PackedConvert,
  kVop3SdstScale,
};

enum class StubExecutionDomain {
  kUnknown,
  kVectorAlu,
  kMatrix,
  kTensorMemory,
  kConversion,
  kScaleAssist,
};

enum class StubOperandLayoutKind {
  kUnknown,
  kPkAddBf16,
  kPkFmaBf16,
  kWmmaF32_16x16x4_F32W32,
  kWmmaLdScalePairedB32,
  kTensorLoadToLds,
  kTensorStoreFromLds,
  kCvtF16Fp8,
  kCvtF32Fp8,
  kVDivScaleF64,
};

struct StubOperandLayoutRecord {
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

enum class StubOperandRole {
  kUnknown,
  kDestination,
  kSource0,
  kSource1,
  kSource2,
  kAccumulator,
  kScale,
  kPairedScale,
  kTensorDescriptor,
  kTensorCoordinate,
  kLdsDestination,
  kLdsSource,
};

struct StubOperandRoleBinding {
  StubOperandRole role = StubOperandRole::kUnknown;
  std::uint32_t count = 0;
  bool is_output = false;
  bool is_implicit = false;
};

struct StubOperandRoleRecord {
  std::array<StubOperandRoleBinding, 6> bindings{};
  std::uint32_t binding_count = 0;
};

struct StubDecodedInstruction {
  std::string_view instruction_name{};
  StubDecodeStatus status = StubDecodeStatus::kUnknownInstruction;
  StubDecoderRoute route = StubDecoderRoute::kUnsupported;
  std::string_view route_name{};
  std::string_view entrypoint_name{};
  std::uint32_t route_priority = 0;
  std::string_view rdna4_encoding_name{};
  std::uint32_t rdna4_opcode = 0;
  std::uint32_t rdna4_operand_count = 0;
  bool appears_in_rdna4_xml = false;
  bool is_target_specific = false;
  StubOpcodeShape opcode_shape = StubOpcodeShape::kUnknown;
  StubExecutionDomain execution_domain = StubExecutionDomain::kUnknown;
  bool uses_accumulator = false;
  bool uses_tensor_memory = false;
  bool uses_scale_path = false;
  bool uses_paired_operands = false;
  StubOperandLayoutRecord operand_layout{};
  StubOperandRoleRecord operand_roles{};
};

struct StubDecoderEntrypointManifest {
  StubDecoderRoute route = StubDecoderRoute::kUnsupported;
  std::string_view route_name{};
  std::string_view entrypoint_name{};
  std::uint32_t route_priority = 0;
  std::uint32_t instruction_count = 0;
};

StubDecodedInstruction DecodeStubInstruction(std::string_view instruction_name);
StubDecodedInstruction DecodeStubInstruction(const StubDecoderRouteInfo& route_info);
StubDecodedInstruction DecodeVop3pStub(std::string_view instruction_name);
StubDecodedInstruction DecodeMimgTensorStub(std::string_view instruction_name);
StubDecodedInstruction DecodeVop1Stub(std::string_view instruction_name);
StubDecodedInstruction DecodeVop3SdstStub(std::string_view instruction_name);
std::string_view GetStubOpcodeShapeName(StubOpcodeShape opcode_shape);
std::string_view GetStubExecutionDomainName(
    StubExecutionDomain execution_domain);
std::string_view GetStubOperandLayoutName(
    StubOperandLayoutKind operand_layout_kind);
std::string_view GetStubOperandRoleName(StubOperandRole operand_role);
std::span<const StubDecoderEntrypointManifest> GetStubDecoderEntrypointManifests();
const StubDecoderEntrypointManifest* FindStubDecoderEntrypointManifest(
    StubDecoderRoute route);

}  // namespace mirage::sim::isa::gfx1250

#endif  // MIRAGE_SIM_ISA_GFX1250_STUB_DECODER_H_
