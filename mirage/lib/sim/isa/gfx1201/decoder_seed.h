#ifndef MIRAGE_SIM_ISA_GFX1201_DECODER_SEED_H_
#define MIRAGE_SIM_ISA_GFX1201_DECODER_SEED_H_

#include <cstdint>
#include <span>
#include <string_view>

#include "lib/sim/isa/gfx1201/support_catalog.h"

namespace mirage::sim::isa {

struct Gfx1201DecoderSeedEntry {
  std::string_view instruction_name;
  std::string_view encoding_name;
  std::string_view encoding_condition;
  std::uint32_t opcode = 0;
  std::uint16_t operand_count = 0;
  Gfx1201SupportRollup rollup = Gfx1201SupportRollup::kGfx1201Specific;
  Gfx1201SupportState state = Gfx1201SupportState::kGfx1201Specific;
  bool is_default_encoding = false;
};

struct Gfx1201DecoderSeedEncoding {
  std::string_view encoding_name;
  std::string_view example_instruction;
  std::string_view rationale;
  std::uint32_t instruction_count = 0;
  std::uint32_t entry_begin = 0;
  std::uint32_t entry_count = 0;
  std::uint32_t default_entry_count = 0;
  std::uint32_t alternate_entry_count = 0;
  std::uint32_t transferable_as_is_count = 0;
  std::uint32_t transferable_with_decoder_work_count = 0;
  std::uint32_t transferable_with_semantic_work_count = 0;
  std::uint32_t transferable_with_decoder_and_semantic_work_count = 0;
  std::uint32_t gfx1201_specific_count = 0;

  constexpr std::uint32_t TransferableWithDecoderWorkRollupCount() const {
    return transferable_with_decoder_work_count +
           transferable_with_decoder_and_semantic_work_count;
  }
};

std::span<const Gfx1201DecoderSeedEncoding>
GetGfx1201Phase0ComputeDecoderSeeds();
const Gfx1201DecoderSeedEncoding* FindGfx1201Phase0ComputeDecoderSeed(
    std::string_view encoding_name);
std::span<const Gfx1201DecoderSeedEntry> GetGfx1201Phase0ComputeDecoderSeedEntries();
std::span<const Gfx1201DecoderSeedEntry> GetGfx1201Phase0ComputeDecoderSeedEntries(
    const Gfx1201DecoderSeedEncoding& encoding);
bool IsGfx1201Phase0ComputeEncoding(std::string_view encoding_name);

}  // namespace mirage::sim::isa

#endif  // MIRAGE_SIM_ISA_GFX1201_DECODER_SEED_H_
