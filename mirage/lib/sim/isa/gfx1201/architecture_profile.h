#ifndef MIRAGE_SIM_ISA_GFX1201_ARCHITECTURE_PROFILE_H_
#define MIRAGE_SIM_ISA_GFX1201_ARCHITECTURE_PROFILE_H_

#include <cstdint>
#include <span>
#include <string_view>

#include "lib/sim/isa/instruction_catalog.h"

namespace mirage::sim::isa {

enum class Gfx1201SupportBucket : std::uint8_t {
  kTransferableFull,
  kTransferableDecodeOnly,
  kTransferableSemanticOnly,
  kKnownButUnsupported,
  kNewVsGfx950,
};

struct Gfx1201SupportBucketSummary {
  Gfx1201SupportBucket bucket = Gfx1201SupportBucket::kNewVsGfx950;
  std::string_view label;
  std::uint32_t instruction_count = 0;
  std::string_view description;
};

struct Gfx1201EncodingFocus {
  std::string_view encoding_name;
  std::uint32_t instruction_count = 0;
  std::string_view example_instruction;
  std::string_view rationale;
};

struct Gfx1201FamilyFocus {
  std::string_view family_name;
  Gfx1201SupportBucket bucket = Gfx1201SupportBucket::kNewVsGfx950;
  std::uint32_t instruction_count = 0;
  std::string_view example_instruction;
  std::string_view rationale;
};

const InstructionCatalogMetadata& GetGfx1201ImportedInstructionMetadata();
std::span<const Gfx1201SupportBucketSummary> GetGfx1201SupportBucketSummaries();
std::span<const Gfx1201EncodingFocus> GetGfx1201Phase0DecoderFocus();
std::span<const Gfx1201EncodingFocus> GetGfx1201Phase1DecoderFocus();
std::span<const Gfx1201FamilyFocus> GetGfx1201CarryOverFamilyFocus();
std::span<const Gfx1201FamilyFocus> GetGfx1201Rdna4DeltaFamilyFocus();
std::string_view ToString(Gfx1201SupportBucket bucket);
std::string_view DescribeGfx1201BringupPhase();

}  // namespace mirage::sim::isa

#endif  // MIRAGE_SIM_ISA_GFX1201_ARCHITECTURE_PROFILE_H_
