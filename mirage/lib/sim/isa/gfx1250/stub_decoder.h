#ifndef MIRAGE_SIM_ISA_GFX1250_STUB_DECODER_H_
#define MIRAGE_SIM_ISA_GFX1250_STUB_DECODER_H_

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
std::span<const StubDecoderEntrypointManifest> GetStubDecoderEntrypointManifests();
const StubDecoderEntrypointManifest* FindStubDecoderEntrypointManifest(
    StubDecoderRoute route);

}  // namespace mirage::sim::isa::gfx1250

#endif  // MIRAGE_SIM_ISA_GFX1250_STUB_DECODER_H_
