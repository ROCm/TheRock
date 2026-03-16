#ifndef MIRAGE_SIM_ISA_GFX1201_PHASE0_WAVE32_STATUS_H_
#define MIRAGE_SIM_ISA_GFX1201_PHASE0_WAVE32_STATUS_H_

#include <cstdint>
#include <span>
#include <string_view>

namespace mirage::sim::isa {

struct Gfx1201Wave32Phase0EncodingStatus {
  std::string_view encoding_name;
  std::uint32_t seeded_instruction_count = 0;
  std::uint32_t executable_instruction_count = 0;
  bool fully_executable = false;
};

struct Gfx1201Wave32Phase0NextRiskEncodingStatus {
  std::string_view encoding_name;
  std::string_view example_instruction;
  std::string_view rationale;
  std::uint32_t seeded_instruction_count = 0;
  std::uint32_t transferable_as_is_count = 0;
  std::uint32_t transferable_with_decoder_work_count = 0;
  std::uint32_t transferable_with_semantic_work_count = 0;
  std::uint32_t transferable_with_decoder_and_semantic_work_count = 0;
  std::uint32_t gfx1201_specific_count = 0;

  constexpr std::uint32_t TransferableWithDecoderRollupCount() const {
    return transferable_with_decoder_work_count +
           transferable_with_decoder_and_semantic_work_count;
  }
};

std::span<const Gfx1201Wave32Phase0EncodingStatus>
GetGfx1201Wave32Phase0EncodingStatuses();
const Gfx1201Wave32Phase0EncodingStatus* FindGfx1201Wave32Phase0EncodingStatus(
    std::string_view encoding_name);
std::span<const Gfx1201Wave32Phase0NextRiskEncodingStatus>
GetGfx1201Wave32Phase0NextRiskEncodingStatuses();
const Gfx1201Wave32Phase0NextRiskEncodingStatus*
FindGfx1201Wave32Phase0NextRiskEncodingStatus(std::string_view encoding_name);
std::span<const std::string_view> GetGfx1201Wave32Phase0NextRiskEncodings();
std::span<const std::string_view> GetGfx1201Wave32Phase0FrontierOrder();
std::string_view GetGfx1201Wave32Phase0RecommendedNextEncoding();
bool IsGfx1201Wave32Phase0EncodingSaturated(std::string_view encoding_name);

}  // namespace mirage::sim::isa

#endif  // MIRAGE_SIM_ISA_GFX1201_PHASE0_WAVE32_STATUS_H_
