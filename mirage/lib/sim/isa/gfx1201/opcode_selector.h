#ifndef MIRAGE_SIM_ISA_GFX1201_OPCODE_SELECTOR_H_
#define MIRAGE_SIM_ISA_GFX1201_OPCODE_SELECTOR_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "lib/sim/isa/gfx1201/decoder_seed.h"

namespace mirage::sim::isa {

enum class Gfx1201OpcodeRouteStatus : std::uint8_t {
  kMatchedSeedEntry,
  kMatchedEncodingOnly,
  kNeedsMoreWords,
  kUnsupportedEncoding,
};

struct Gfx1201OpcodeSelectorRule {
  std::string_view encoding_name;
  std::uint8_t instruction_dword_count = 0;
  std::uint8_t opcode_bit_offset = 0;
  std::uint8_t opcode_bit_width = 0;
  std::string_view selector_description;
};

struct Gfx1201OpcodeRoute {
  Gfx1201OpcodeRouteStatus status = Gfx1201OpcodeRouteStatus::kUnsupportedEncoding;
  const Gfx1201OpcodeSelectorRule* selector_rule = nullptr;
  const Gfx1201DecoderSeedEncoding* seed_encoding = nullptr;
  const Gfx1201DecoderSeedEntry* seed_entry = nullptr;
  std::uint32_t opcode = 0;
  std::size_t words_required = 0;

  constexpr bool HasEncoding() const { return selector_rule != nullptr; }
  constexpr bool HasSeedEntry() const { return seed_entry != nullptr; }
};

std::span<const Gfx1201OpcodeSelectorRule>
GetGfx1201Phase0ComputeOpcodeSelectorRules();
const Gfx1201OpcodeSelectorRule* FindGfx1201Phase0ComputeOpcodeSelectorRule(
    std::string_view encoding_name);
bool SelectGfx1201Phase0ComputeOpcodeRoute(
    std::span<const std::uint32_t> words,
    Gfx1201OpcodeRoute* route,
    std::string* error_message = nullptr);
std::string_view ToString(Gfx1201OpcodeRouteStatus status);

}  // namespace mirage::sim::isa

#endif  // MIRAGE_SIM_ISA_GFX1201_OPCODE_SELECTOR_H_
