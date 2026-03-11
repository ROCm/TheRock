#ifndef MIRAGE_SIM_ISA_GFX1250_STUB_DECODER_H_
#define MIRAGE_SIM_ISA_GFX1250_STUB_DECODER_H_

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

#include "lib/sim/isa/common/operand_metadata.h"
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
  kPkMulBf16,
  kPkMinNumBf16,
  kPkMaxNumBf16,
  kWmmaF32_16x16x4_F32W32,
  kWmmaF32_16x16x128_Fp8Fp8W32,
  kWmmaF16_16x16x128_Fp8Fp8W32,
  kWmmaF32_16x16x64_Fp8Fp8W32,
  kWmmaCoreGeneric,
  kWmmaScaleF32_16x16x128_F8F6F4,
  kWmmaScale16F32_16x16x128_F8F6F4,
  kWmmaScaleGeneric,
  kWmmaLdScalePairedB32,
  kWmmaLdScale16PairedB64,
  kSwmmacF32_16x16x128_Fp8Fp8W32,
  kSwmmacF16_16x16x128_Fp8Fp8W32,
  kSwmmacCoreGeneric,
  kTensorLoadToLds,
  kTensorStoreFromLds,
  kCvtF16Bf8,
  kCvtF16Fp8,
  kCvtF32Fp8,
  kCvtPkF16Fp8,
  kCvtPkF16Bf8,
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

using StubFragmentKind = FragmentKind;
using StubFragmentShape = FragmentShape;
using StubOperandAccess = OperandAccess;
using StubOperandRole = OperandRole;
using StubOperandRoleBinding = OperandRoleBinding;
using StubOperandRoleRecord = OperandRoleRecord;
using StubOperandSlotKind = OperandSlotKind;
using StubOperandValueClass = OperandValueClass;
using StubOperandDescriptor = OperandDescriptor;
using StubOperandSlotBinding = OperandSlotBinding;
using StubOperandSlotRecord = OperandSlotRecord;

struct StubOperandDescriptorRecord {
  std::array<StubOperandDescriptor, 8> descriptors{};
  std::uint32_t descriptor_count = 0;
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
  StubOperandSlotRecord operand_slots{};
  StubOperandDescriptorRecord operand_descriptors{};
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
std::string_view GetStubOperandSlotKindName(
    StubOperandSlotKind operand_slot_kind);
std::string_view GetStubOperandValueClassName(
    StubOperandValueClass operand_value_class);
std::span<const StubDecoderEntrypointManifest> GetStubDecoderEntrypointManifests();
const StubDecoderEntrypointManifest* FindStubDecoderEntrypointManifest(
    StubDecoderRoute route);

}  // namespace mirage::sim::isa::gfx1250

#endif  // MIRAGE_SIM_ISA_GFX1250_STUB_DECODER_H_
