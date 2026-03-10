#include "lib/sim/isa/gfx1201/opcode_selector.h"

#include <array>

namespace mirage::sim::isa {
namespace {

constexpr std::uint32_t kEncSopp = 0x17f;
constexpr std::uint32_t kEncSopc = 0x17e;
constexpr std::uint32_t kEncSop1 = 0x17d;
constexpr std::uint32_t kEncSop2 = 0x2;
constexpr std::uint32_t kEncSopk = 0xb;
constexpr std::uint32_t kEncSmem = 0x30;
constexpr std::uint32_t kEncVop1 = 0x3f;
constexpr std::uint32_t kEncVopc = 0x3e;
constexpr std::uint32_t kEncVop2 = 0x0;
constexpr std::uint32_t kEncVop3 = 0x34;
constexpr std::uint32_t kEncVds = 0x36;
constexpr std::uint32_t kEncVglobalWord = 55;

constexpr std::uint32_t ExtractBits(std::uint32_t value,
                                    std::uint32_t bit_offset,
                                    std::uint32_t bit_count) {
  if (bit_count == 32) {
    return value;
  }
  return (value >> bit_offset) & ((1u << bit_count) - 1u);
}

constexpr bool IsPhase0VglobalWord(std::uint32_t word) {
  return ExtractBits(word, 26, 6) == kEncVglobalWord &&
         ExtractBits(word, 14, 2) == 2u;
}

constexpr std::array<Gfx1201OpcodeSelectorRule, 12> kPhase0ComputeRules{{
    {"ENC_VGLOBAL", 2u, 18u, 7u, "bits[26:31]==55 and bits[14:15]==2"},
    {"ENC_VDS", 2u, 17u, 8u, "bits[26:31]==0x36"},
    {"ENC_SMEM", 2u, 18u, 8u, "bits[26:31]==0x30"},
    {"ENC_VOP3", 2u, 16u, 10u, "bits[26:31]==0x34"},
    {"ENC_SOPP", 1u, 16u, 7u, "bits[23:31]==0x17f"},
    {"ENC_SOPC", 1u, 16u, 7u, "bits[23:31]==0x17e"},
    {"ENC_SOP1", 1u, 8u, 8u, "bits[23:31]==0x17d"},
    {"ENC_SOPK", 1u, 23u, 5u, "bits[28:31]==0xb"},
    {"ENC_SOP2", 1u, 23u, 7u, "bits[30:31]==0x2"},
    {"ENC_VOP1", 1u, 9u, 8u, "bits[25:31]==0x3f"},
    {"ENC_VOPC", 1u, 17u, 8u, "bits[25:31]==0x3e"},
    {"ENC_VOP2", 1u, 25u, 6u, "bit[31]==0"},
}};

const Gfx1201DecoderSeedEntry* FindSeedEntryByOpcode(
    const Gfx1201DecoderSeedEncoding& seed_encoding,
    std::uint32_t opcode) {
  for (const Gfx1201DecoderSeedEntry& entry :
       GetGfx1201Phase0ComputeDecoderSeedEntries(seed_encoding)) {
    if (entry.opcode == opcode) {
      return &entry;
    }
  }
  return nullptr;
}

bool MatchPhase0Rule(std::uint32_t word, const Gfx1201OpcodeSelectorRule** rule) {
  if (rule == nullptr) {
    return false;
  }

  if (IsPhase0VglobalWord(word)) {
    *rule = &kPhase0ComputeRules[0];
    return true;
  }
  if (ExtractBits(word, 26, 6) == kEncVds) {
    *rule = &kPhase0ComputeRules[1];
    return true;
  }
  if (ExtractBits(word, 26, 6) == kEncSmem) {
    *rule = &kPhase0ComputeRules[2];
    return true;
  }
  if (ExtractBits(word, 26, 6) == kEncVop3) {
    *rule = &kPhase0ComputeRules[3];
    return true;
  }
  if (ExtractBits(word, 23, 9) == kEncSopp) {
    *rule = &kPhase0ComputeRules[4];
    return true;
  }
  if (ExtractBits(word, 23, 9) == kEncSopc) {
    *rule = &kPhase0ComputeRules[5];
    return true;
  }
  if (ExtractBits(word, 23, 9) == kEncSop1) {
    *rule = &kPhase0ComputeRules[6];
    return true;
  }
  if (ExtractBits(word, 28, 4) == kEncSopk) {
    *rule = &kPhase0ComputeRules[7];
    return true;
  }
  if (ExtractBits(word, 30, 2) == kEncSop2) {
    *rule = &kPhase0ComputeRules[8];
    return true;
  }
  if (ExtractBits(word, 25, 7) == kEncVop1) {
    *rule = &kPhase0ComputeRules[9];
    return true;
  }
  if (ExtractBits(word, 25, 7) == kEncVopc) {
    *rule = &kPhase0ComputeRules[10];
    return true;
  }
  if (ExtractBits(word, 31, 1) == kEncVop2) {
    *rule = &kPhase0ComputeRules[11];
    return true;
  }
  return false;
}

}  // namespace

std::span<const Gfx1201OpcodeSelectorRule>
GetGfx1201Phase0ComputeOpcodeSelectorRules() {
  return kPhase0ComputeRules;
}

const Gfx1201OpcodeSelectorRule* FindGfx1201Phase0ComputeOpcodeSelectorRule(
    std::string_view encoding_name) {
  for (const Gfx1201OpcodeSelectorRule& rule : kPhase0ComputeRules) {
    if (rule.encoding_name == encoding_name) {
      return &rule;
    }
  }
  return nullptr;
}

bool SelectGfx1201Phase0ComputeOpcodeRoute(
    std::span<const std::uint32_t> words,
    Gfx1201OpcodeRoute* route,
    std::string* error_message) {
  if (route == nullptr) {
    if (error_message != nullptr) {
      *error_message = "opcode route output must not be null";
    }
    return false;
  }

  *route = Gfx1201OpcodeRoute{};
  if (words.empty()) {
    if (error_message != nullptr) {
      *error_message = "instruction stream is empty";
    }
    return false;
  }

  const Gfx1201OpcodeSelectorRule* rule = nullptr;
  if (!MatchPhase0Rule(words.front(), &rule)) {
    route->status = Gfx1201OpcodeRouteStatus::kUnsupportedEncoding;
    if (error_message != nullptr) {
      *error_message = "unsupported or unknown gfx1201 phase-0 compute encoding";
    }
    return false;
  }

  route->selector_rule = rule;
  route->seed_encoding = FindGfx1201Phase0ComputeDecoderSeed(rule->encoding_name);
  route->words_required = rule->instruction_dword_count;
  route->opcode =
      ExtractBits(words.front(), rule->opcode_bit_offset, rule->opcode_bit_width);

  if (words.size() < route->words_required) {
    route->status = Gfx1201OpcodeRouteStatus::kNeedsMoreWords;
    return true;
  }

  if (route->seed_encoding == nullptr) {
    route->status = Gfx1201OpcodeRouteStatus::kUnsupportedEncoding;
    if (error_message != nullptr) {
      *error_message = "matched phase-0 selector rule without decoder seed";
    }
    return false;
  }

  route->seed_entry = FindSeedEntryByOpcode(*route->seed_encoding, route->opcode);
  route->status = route->seed_entry != nullptr
                      ? Gfx1201OpcodeRouteStatus::kMatchedSeedEntry
                      : Gfx1201OpcodeRouteStatus::kMatchedEncodingOnly;
  return true;
}

std::string_view ToString(Gfx1201OpcodeRouteStatus status) {
  switch (status) {
    case Gfx1201OpcodeRouteStatus::kMatchedSeedEntry:
      return "matched_seed_entry";
    case Gfx1201OpcodeRouteStatus::kMatchedEncodingOnly:
      return "matched_encoding_only";
    case Gfx1201OpcodeRouteStatus::kNeedsMoreWords:
      return "needs_more_words";
    case Gfx1201OpcodeRouteStatus::kUnsupportedEncoding:
      return "unsupported_encoding";
  }
  return "unknown";
}

}  // namespace mirage::sim::isa
