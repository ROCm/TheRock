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

std::span<const Gfx1201Wave32Phase0EncodingStatus>
GetGfx1201Wave32Phase0EncodingStatuses();
const Gfx1201Wave32Phase0EncodingStatus* FindGfx1201Wave32Phase0EncodingStatus(
    std::string_view encoding_name);
std::span<const std::string_view> GetGfx1201Wave32Phase0NextRiskEncodings();
bool IsGfx1201Wave32Phase0EncodingSaturated(std::string_view encoding_name);

}  // namespace mirage::sim::isa

#endif  // MIRAGE_SIM_ISA_GFX1201_PHASE0_WAVE32_STATUS_H_
