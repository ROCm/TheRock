#ifndef MIRAGE_SIM_ISA_GFX1201_SUPPORT_CATALOG_H_
#define MIRAGE_SIM_ISA_GFX1201_SUPPORT_CATALOG_H_

#include <cstdint>
#include <span>
#include <string_view>

#include "lib/sim/isa/instruction_catalog.h"

namespace mirage::sim::isa {

enum class Gfx1201SupportRollup : std::uint8_t {
  kTransferableAsIs,
  kTransferableWithDecoderWork,
  kTransferableWithSemanticWork,
  kGfx1201Specific,
};

enum class Gfx1201SupportState : std::uint8_t {
  kTransferableAsIs,
  kTransferableWithDecoderWork,
  kTransferableWithSemanticWork,
  kTransferableWithDecoderAndSemanticWork,
  kGfx1201Specific,
};

struct Gfx1201InstructionSupportInfo {
  std::string_view instruction_name;
  Gfx1201SupportRollup rollup = Gfx1201SupportRollup::kGfx1201Specific;
  Gfx1201SupportState state = Gfx1201SupportState::kGfx1201Specific;
  InstructionFlags flags;
  bool known_in_gfx950_catalog = false;
  bool decoder_supported_in_gfx950 = false;
  bool semantic_supported_in_gfx950 = false;
  std::uint32_t encoding_begin = 0;
  std::uint16_t encoding_count = 0;
};

struct Gfx1201SupportSummary {
  Gfx1201SupportRollup rollup = Gfx1201SupportRollup::kGfx1201Specific;
  std::uint32_t instruction_count = 0;
  std::string_view description;
};

struct Gfx1201SupportStateSummary {
  Gfx1201SupportState state = Gfx1201SupportState::kGfx1201Specific;
  std::uint32_t instruction_count = 0;
  std::string_view description;
};

constexpr bool IsTransferableFromGfx950(
    const Gfx1201InstructionSupportInfo& instruction) {
  return instruction.rollup != Gfx1201SupportRollup::kGfx1201Specific;
}

constexpr bool NeedsDecoderWork(
    const Gfx1201InstructionSupportInfo& instruction) {
  return !instruction.decoder_supported_in_gfx950;
}

constexpr bool NeedsSemanticWork(
    const Gfx1201InstructionSupportInfo& instruction) {
  return !instruction.semantic_supported_in_gfx950;
}

const InstructionCatalogMetadata& GetGfx1201SupportCatalogMetadata();
std::span<const Gfx1201InstructionSupportInfo>
GetGfx1201InstructionSupportCatalog();
std::span<const InstructionEncodingSpec> GetGfx1201InstructionSupportEncodings();
const Gfx1201InstructionSupportInfo* FindGfx1201InstructionSupport(
    std::string_view instruction_name);
std::span<const Gfx1201InstructionSupportInfo> GetGfx1201InstructionsByRollup(
    Gfx1201SupportRollup rollup);
std::span<const Gfx1201InstructionSupportInfo> GetGfx1201InstructionsByState(
    Gfx1201SupportState state);
std::span<const InstructionEncodingSpec> GetGfx1201Encodings(
    const Gfx1201InstructionSupportInfo& instruction);
std::span<const Gfx1201SupportSummary> GetGfx1201SupportRollupSummaries();
std::span<const Gfx1201SupportStateSummary> GetGfx1201SupportStateSummaries();
std::string_view ToString(Gfx1201SupportRollup rollup);
std::string_view ToString(Gfx1201SupportState state);

}  // namespace mirage::sim::isa

#endif  // MIRAGE_SIM_ISA_GFX1201_SUPPORT_CATALOG_H_
