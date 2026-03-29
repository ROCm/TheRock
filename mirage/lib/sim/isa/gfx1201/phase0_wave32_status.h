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
  std::uint32_t executable_instruction_count = 0;
  std::string_view first_executable_instruction;
  std::uint32_t transferable_as_is_count = 0;
  std::uint32_t transferable_with_decoder_work_count = 0;
  std::uint32_t transferable_with_semantic_work_count = 0;
  std::uint32_t transferable_with_decoder_and_semantic_work_count = 0;
  std::uint32_t gfx1201_specific_count = 0;

  constexpr bool HasExecutableFoothold() const {
    return executable_instruction_count != 0;
  }

  constexpr std::uint32_t TransferableWithDecoderRollupCount() const {
    return transferable_with_decoder_work_count +
           transferable_with_decoder_and_semantic_work_count;
  }
};

struct Gfx1201Wave32Phase0VdsBoundaryBucket {
  std::string_view bucket_name;
  std::string_view example_instruction;
  std::string_view rationale;
  std::string_view blocking_dimension;
  std::uint32_t risk_rank = 0;
  std::uint32_t starting_instruction_ordinal = 0;
  std::uint32_t ending_instruction_ordinal = 0;
  std::uint32_t instruction_count = 0;
  bool safe_under_current_request = false;
  std::span<const std::string_view> instruction_names;
};

struct Gfx1201Wave32Phase0VdsBoundaryInstructionStatus {
  std::string_view instruction_name;
  std::string_view bucket_name;
  std::string_view blocking_dimension;
  std::uint32_t bucket_risk_rank = 0;
  std::uint32_t tail_ordinal = 0;
  std::uint32_t bucket_ordinal = 0;
  std::uint32_t opcode = 0;
  std::uint16_t operand_count = 0;
  std::string_view support_rollup;
  std::string_view support_state;
  bool safe_under_current_request = false;
};

struct Gfx1201Wave32Phase0VdsBoundaryBucketStatus {
  std::string_view bucket_name;
  std::string_view blocking_dimension;
  std::uint32_t risk_rank = 0;
  std::uint32_t instruction_count = 0;
  std::uint32_t first_opcode = 0;
  std::uint32_t last_opcode = 0;
  std::uint16_t min_operand_count = 0;
  std::uint16_t max_operand_count = 0;
  std::uint32_t operand_count_3_count = 0;
  std::uint32_t operand_count_4_count = 0;
  std::uint32_t operand_count_5_count = 0;
  std::uint32_t operand_count_6_count = 0;
  std::uint32_t transferable_with_decoder_work_count = 0;
  std::uint32_t transferable_with_decoder_and_semantic_work_count = 0;
  std::uint32_t gfx1201_specific_count = 0;
  std::uint32_t transferable_as_is_state_count = 0;
  std::uint32_t transferable_with_decoder_work_state_count = 0;
  std::uint32_t transferable_with_semantic_work_state_count = 0;
  std::uint32_t transferable_with_decoder_and_semantic_work_state_count = 0;
  std::uint32_t gfx1201_specific_state_count = 0;
  bool safe_under_current_request = false;
};

struct Gfx1201Wave32Phase0VdsNextRiskStep {
  std::string_view bucket_name;
  std::string_view blocking_dimension;
  std::uint32_t risk_rank = 0;
  std::string_view first_instruction_name;
  std::string_view last_instruction_name;
  std::uint32_t instruction_count = 0;
  std::uint32_t remaining_instruction_count_including_bucket = 0;
  std::uint32_t remaining_instruction_count_after_bucket = 0;
  std::string_view next_bucket_name;
  std::string_view next_bucket_blocking_dimension;
  std::string_view next_instruction_name;
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
std::span<const Gfx1201Wave32Phase0VdsBoundaryBucket>
GetGfx1201Wave32Phase0VdsBoundaryBuckets();
std::span<const Gfx1201Wave32Phase0VdsBoundaryBucketStatus>
GetGfx1201Wave32Phase0VdsBoundaryBucketStatuses();
std::span<const std::string_view> GetGfx1201Wave32Phase0VdsBoundaryOrder();
std::span<const Gfx1201Wave32Phase0VdsBoundaryInstructionStatus>
GetGfx1201Wave32Phase0RemainingVdsInstructionStatuses();
std::span<const Gfx1201Wave32Phase0VdsNextRiskStep>
GetGfx1201Wave32Phase0VdsNextRiskSteps();
const Gfx1201Wave32Phase0VdsBoundaryBucket*
FindGfx1201Wave32Phase0VdsBoundaryBucket(std::string_view bucket_name);
const Gfx1201Wave32Phase0VdsBoundaryBucketStatus*
FindGfx1201Wave32Phase0VdsBoundaryBucketStatus(std::string_view bucket_name);
const Gfx1201Wave32Phase0VdsBoundaryBucket*
FindGfx1201Wave32Phase0VdsBoundaryBucketForInstruction(
    std::string_view instruction_name);
const Gfx1201Wave32Phase0VdsBoundaryInstructionStatus*
FindGfx1201Wave32Phase0RemainingVdsInstructionStatus(
    std::string_view instruction_name);
const Gfx1201Wave32Phase0VdsBoundaryInstructionStatus*
FindGfx1201Wave32Phase0RemainingVdsInstructionStatusByOpcode(
    std::uint32_t opcode);
const Gfx1201Wave32Phase0VdsNextRiskStep*
FindGfx1201Wave32Phase0VdsNextRiskStep(std::string_view bucket_name);
bool HasGfx1201Wave32SafeVdsContinuation();
std::string_view GetGfx1201Wave32RecommendedNextVdsBucket();
std::string_view GetGfx1201Wave32FirstUnsafeVdsBucket();
std::string_view GetGfx1201Wave32FirstUnsafeVdsBlockingDimension();
std::span<const std::string_view> GetGfx1201Wave32FirstUnsafeVdsInstructions();
std::string_view GetGfx1201Wave32Phase0RecommendedNextEncoding();
bool IsGfx1201Wave32Phase0EncodingSaturated(std::string_view encoding_name);

}  // namespace mirage::sim::isa

#endif  // MIRAGE_SIM_ISA_GFX1201_PHASE0_WAVE32_STATUS_H_
