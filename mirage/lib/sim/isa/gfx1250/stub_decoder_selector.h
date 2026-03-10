#ifndef MIRAGE_SIM_ISA_GFX1250_STUB_DECODER_SELECTOR_H_
#define MIRAGE_SIM_ISA_GFX1250_STUB_DECODER_SELECTOR_H_

#include <cstdint>
#include <span>
#include <string_view>

#include "lib/sim/isa/gfx1250/decoder_seed_catalog.h"

namespace mirage::sim::isa::gfx1250 {

enum class StubDecoderRoute {
  kUnsupported,
  kVop3p,
  kMimgTensor,
  kVop1,
  kVop3Sdst,
};

struct StubDecoderRouteInfo {
  std::string_view instruction_name{};
  StubDecoderRoute route = StubDecoderRoute::kUnsupported;
  std::string_view route_name{};
  std::uint32_t route_priority = 0;
  DecodeSeedHint decode_hint = DecodeSeedHint::kUnknown;
  std::string_view rdna4_encoding_name{};
  std::uint32_t rdna4_opcode = 0;
  std::uint32_t rdna4_operand_count = 0;
  bool appears_in_rdna4_xml = false;
  bool is_target_specific = false;
};

struct StubDecoderRouteManifest {
  StubDecoderRoute route = StubDecoderRoute::kUnsupported;
  std::string_view route_name{};
  std::uint32_t route_priority = 0;
  std::uint32_t instruction_count = 0;
  std::uint32_t xml_backed_count = 0;
  std::uint32_t llvm_only_count = 0;
  std::uint32_t target_specific_count = 0;
};

StubDecoderRoute SelectStubDecoderRoute(DecodeSeedHint decode_hint);
StubDecoderRoute SelectStubDecoderRoute(std::string_view instruction_name);
std::string_view GetStubDecoderRouteName(StubDecoderRoute route);
std::span<const StubDecoderRouteInfo> GetStubDecoderRouteInfos();
const StubDecoderRouteInfo* FindStubDecoderRouteInfo(std::string_view instruction_name);
std::span<const std::string_view> GetStubDecoderRouteInstructions(
    StubDecoderRoute route);
std::span<const StubDecoderRouteManifest> GetStubDecoderRouteManifests();
const StubDecoderRouteManifest* FindStubDecoderRouteManifest(
    StubDecoderRoute route);

}  // namespace mirage::sim::isa::gfx1250

#endif  // MIRAGE_SIM_ISA_GFX1250_STUB_DECODER_SELECTOR_H_
