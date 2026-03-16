#include "lib/sim/isa/gfx1201/phase0_wave32_status.h"

#include <algorithm>
#include <array>
#include <vector>

#include "lib/sim/isa/gfx1201/binary_decoder.h"
#include "lib/sim/isa/gfx1201/decoder_seed.h"

namespace mirage::sim::isa {
namespace {

constexpr std::array<std::string_view, 3> kTrackedEncodings{{
    "ENC_VOP1",
    "ENC_VOP2",
    "ENC_VOPC",
}};

constexpr std::array<std::string_view, 4> kNextRiskEncodings{{
    "ENC_SMEM",
    "ENC_VOP3",
    "ENC_VDS",
    "ENC_VGLOBAL",
}};

constexpr std::array<std::string_view, 4> kFrontierOrder{{
    "ENC_SMEM",
    "ENC_VGLOBAL",
    "ENC_VDS",
    "ENC_VOP3",
}};

std::uint32_t CountExecutableInstructions(const Gfx1201DecoderSeedEncoding& encoding,
                                          const Gfx1201BinaryDecoder& decoder) {
  std::vector<std::string_view> seen_instruction_names;
  seen_instruction_names.reserve(encoding.instruction_count);

  std::uint32_t executable_instruction_count = 0;
  for (const Gfx1201DecoderSeedEntry& entry :
       GetGfx1201Phase0ComputeDecoderSeedEntries(encoding)) {
    if (std::find(seen_instruction_names.begin(), seen_instruction_names.end(),
                  entry.instruction_name) != seen_instruction_names.end()) {
      continue;
    }
    seen_instruction_names.push_back(entry.instruction_name);
    if (decoder.SupportsPhase0ExecutableOpcode(entry.instruction_name)) {
      ++executable_instruction_count;
    }
  }

  return executable_instruction_count;
}

std::array<Gfx1201Wave32Phase0EncodingStatus, kTrackedEncodings.size()>
BuildStatuses() {
  Gfx1201BinaryDecoder decoder;
  std::array<Gfx1201Wave32Phase0EncodingStatus, kTrackedEncodings.size()>
      statuses{};

  for (std::size_t i = 0; i < kTrackedEncodings.size(); ++i) {
    const Gfx1201DecoderSeedEncoding* seed =
        FindGfx1201Phase0ComputeDecoderSeed(kTrackedEncodings[i]);
    if (seed == nullptr) {
      continue;
    }

    const std::uint32_t executable_instruction_count =
        CountExecutableInstructions(*seed, decoder);
    statuses[i] = Gfx1201Wave32Phase0EncodingStatus{
        seed->encoding_name,
        seed->instruction_count,
        executable_instruction_count,
        executable_instruction_count == seed->instruction_count,
    };
  }

  return statuses;
}

std::array<Gfx1201Wave32Phase0NextRiskEncodingStatus, kNextRiskEncodings.size()>
BuildNextRiskStatuses() {
  std::array<Gfx1201Wave32Phase0NextRiskEncodingStatus, kNextRiskEncodings.size()>
      statuses{};

  for (std::size_t i = 0; i < kNextRiskEncodings.size(); ++i) {
    const Gfx1201DecoderSeedEncoding* seed =
        FindGfx1201Phase0ComputeDecoderSeed(kNextRiskEncodings[i]);
    if (seed == nullptr) {
      continue;
    }

    statuses[i] = Gfx1201Wave32Phase0NextRiskEncodingStatus{
        seed->encoding_name,
        seed->example_instruction,
        seed->rationale,
        seed->instruction_count,
        seed->transferable_as_is_count,
        seed->transferable_with_decoder_work_count,
        seed->transferable_with_semantic_work_count,
        seed->transferable_with_decoder_and_semantic_work_count,
        seed->gfx1201_specific_count,
    };
  }

  return statuses;
}

}  // namespace

std::span<const Gfx1201Wave32Phase0EncodingStatus>
GetGfx1201Wave32Phase0EncodingStatuses() {
  static const auto kStatuses = BuildStatuses();
  return kStatuses;
}

const Gfx1201Wave32Phase0EncodingStatus* FindGfx1201Wave32Phase0EncodingStatus(
    std::string_view encoding_name) {
  for (const Gfx1201Wave32Phase0EncodingStatus& status :
       GetGfx1201Wave32Phase0EncodingStatuses()) {
    if (status.encoding_name == encoding_name) {
      return &status;
    }
  }
  return nullptr;
}

std::span<const Gfx1201Wave32Phase0NextRiskEncodingStatus>
GetGfx1201Wave32Phase0NextRiskEncodingStatuses() {
  static const auto kStatuses = BuildNextRiskStatuses();
  return kStatuses;
}

const Gfx1201Wave32Phase0NextRiskEncodingStatus*
FindGfx1201Wave32Phase0NextRiskEncodingStatus(
    std::string_view encoding_name) {
  for (const Gfx1201Wave32Phase0NextRiskEncodingStatus& status :
       GetGfx1201Wave32Phase0NextRiskEncodingStatuses()) {
    if (status.encoding_name == encoding_name) {
      return &status;
    }
  }
  return nullptr;
}

std::span<const std::string_view> GetGfx1201Wave32Phase0NextRiskEncodings() {
  return kNextRiskEncodings;
}

std::span<const std::string_view> GetGfx1201Wave32Phase0FrontierOrder() {
  return kFrontierOrder;
}

std::string_view GetGfx1201Wave32Phase0RecommendedNextEncoding() {
  return kFrontierOrder.front();
}

bool IsGfx1201Wave32Phase0EncodingSaturated(std::string_view encoding_name) {
  const Gfx1201Wave32Phase0EncodingStatus* status =
      FindGfx1201Wave32Phase0EncodingStatus(encoding_name);
  return status != nullptr && status->fully_executable;
}

}  // namespace mirage::sim::isa
