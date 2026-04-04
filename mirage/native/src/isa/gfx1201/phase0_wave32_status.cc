#include "lib/sim/isa/gfx1201/phase0_wave32_status.h"

#include <algorithm>
#include <array>
#include <vector>

#include "lib/sim/isa/gfx1201/binary_decoder.h"
#include "lib/sim/isa/gfx1201/decoder_seed.h"

namespace mirage::sim::isa {
namespace {

constexpr std::array<std::string_view, 5> kTrackedEncodings{{
    "ENC_VOP1",
    "ENC_VOP2",
    "ENC_VOPC",
    "ENC_SMEM",
    "ENC_VGLOBAL",
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

constexpr std::array<std::string_view, 4> kVdsBoundaryOrder{{
    "append_consume",
    "exchange_compare_store",
    "multi_address",
    "bvh_stack",
}};

constexpr std::array<std::string_view, 2> kAppendConsumeVdsInstructions{{
    "DS_APPEND",
    "DS_CONSUME",
}};

constexpr std::array<std::string_view, 7>
    kExchangeCompareStoreVdsInstructions{{
        "DS_CONDXCHG32_RTN_B64",
        "DS_CMPSTORE_B32",
        "DS_CMPSTORE_B64",
        "DS_CMPSTORE_RTN_B32",
        "DS_CMPSTORE_RTN_B64",
        "DS_STOREXCHG_RTN_B32",
        "DS_STOREXCHG_RTN_B64",
    }};

constexpr std::array<std::string_view, 12> kMultiAddressVdsInstructions{{
    "DS_LOAD_2ADDR_B32",
    "DS_LOAD_2ADDR_B64",
    "DS_LOAD_2ADDR_STRIDE64_B32",
    "DS_LOAD_2ADDR_STRIDE64_B64",
    "DS_STOREXCHG_2ADDR_RTN_B32",
    "DS_STOREXCHG_2ADDR_RTN_B64",
    "DS_STOREXCHG_2ADDR_STRIDE64_RTN_B32",
    "DS_STOREXCHG_2ADDR_STRIDE64_RTN_B64",
    "DS_STORE_2ADDR_B32",
    "DS_STORE_2ADDR_B64",
    "DS_STORE_2ADDR_STRIDE64_B32",
    "DS_STORE_2ADDR_STRIDE64_B64",
}};

constexpr std::array<std::string_view, 3> kBvhStackVdsInstructions{{
    "DS_BVH_STACK_PUSH4_POP1_RTN_B32",
    "DS_BVH_STACK_PUSH8_POP1_RTN_B32",
    "DS_BVH_STACK_PUSH8_POP2_RTN_B64",
}};

constexpr std::array<Gfx1201Wave32Phase0VdsBoundaryBucket, 4>
    kVdsBoundaryBuckets{{
        {
            "append_consume",
            "DS_APPEND",
            "Append/consume allocator semantics are the first remaining non-local LDS tail.",
            "allocator_or_gds_semantics",
            0,
            0,
            1,
            2,
            false,
            std::span<const std::string_view>(kAppendConsumeVdsInstructions),
        },
        {
            "exchange_compare_store",
            "DS_CONDXCHG32_RTN_B64",
            "Exchange and compare-store forms are the next semantic-risk step on the VDS path.",
            "exchange_compare_store_semantics",
            1,
            2,
            8,
            7,
            false,
            std::span<const std::string_view>(kExchangeCompareStoreVdsInstructions),
        },
        {
            "multi_address",
            "DS_LOAD_2ADDR_B32",
            "Remaining two-address and stride64 LDS forms would widen the current one-address execution model.",
            "multi_address_semantics",
            2,
            9,
            20,
            12,
            false,
            std::span<const std::string_view>(kMultiAddressVdsInstructions),
        },
        {
            "bvh_stack",
            "DS_BVH_STACK_PUSH4_POP1_RTN_B32",
            "BVH stack instructions are gfx1201-specific and sit outside the current LDS utility model.",
            "gfx1201_specific_bvh_semantics",
            3,
            21,
            23,
            3,
            false,
            std::span<const std::string_view>(kBvhStackVdsInstructions),
        },
    }};

struct ExecutableInstructionSummary {
  std::uint32_t executable_instruction_count = 0;
  std::string_view first_executable_instruction;
};

struct BucketOpcodePoint {
  std::uint32_t opcode = 0;
  std::string_view instruction_name;
};

const Gfx1201DecoderSeedEntry* FindVdsSeedEntry(std::string_view instruction_name) {
  const Gfx1201DecoderSeedEncoding* vds_seed =
      FindGfx1201Phase0ComputeDecoderSeed("ENC_VDS");
  if (vds_seed == nullptr) {
    return nullptr;
  }
  for (const Gfx1201DecoderSeedEntry& entry :
       GetGfx1201Phase0ComputeDecoderSeedEntries(*vds_seed)) {
    if (entry.instruction_name == instruction_name && entry.is_default_encoding) {
      return &entry;
    }
  }
  return nullptr;
}

ExecutableInstructionSummary SummarizeExecutableInstructions(
    const Gfx1201DecoderSeedEncoding& encoding, const Gfx1201BinaryDecoder& decoder) {
  std::vector<std::string_view> seen_instruction_names;
  seen_instruction_names.reserve(encoding.instruction_count);

  ExecutableInstructionSummary summary;
  for (const Gfx1201DecoderSeedEntry& entry :
       GetGfx1201Phase0ComputeDecoderSeedEntries(encoding)) {
    if (std::find(seen_instruction_names.begin(), seen_instruction_names.end(),
                  entry.instruction_name) != seen_instruction_names.end()) {
      continue;
    }
    seen_instruction_names.push_back(entry.instruction_name);
    if (decoder.SupportsPhase0ExecutableOpcode(entry.instruction_name)) {
      if (summary.executable_instruction_count == 0) {
        summary.first_executable_instruction = entry.instruction_name;
      }
      ++summary.executable_instruction_count;
    }
  }

  return summary;
}

std::vector<BucketOpcodePoint> CollectBucketOpcodePoints(
    const Gfx1201Wave32Phase0VdsBoundaryBucket& bucket) {
  std::vector<BucketOpcodePoint> points;
  points.reserve(bucket.instruction_names.size());

  for (std::string_view instruction_name : bucket.instruction_names) {
    const Gfx1201DecoderSeedEntry* seed_entry = FindVdsSeedEntry(instruction_name);
    if (seed_entry == nullptr) {
      continue;
    }
    points.push_back(BucketOpcodePoint{seed_entry->opcode, instruction_name});
  }

  std::sort(points.begin(), points.end(),
            [](const BucketOpcodePoint& lhs, const BucketOpcodePoint& rhs) {
              if (lhs.opcode != rhs.opcode) {
                return lhs.opcode < rhs.opcode;
              }
              return lhs.instruction_name < rhs.instruction_name;
            });
  return points;
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

    const ExecutableInstructionSummary executable_summary =
        SummarizeExecutableInstructions(*seed, decoder);
    statuses[i] = Gfx1201Wave32Phase0EncodingStatus{
        seed->encoding_name,
        seed->instruction_count,
        executable_summary.executable_instruction_count,
        executable_summary.executable_instruction_count == seed->instruction_count,
    };
  }

  return statuses;
}

std::array<Gfx1201Wave32Phase0NextRiskEncodingStatus, kNextRiskEncodings.size()>
BuildNextRiskStatuses() {
  Gfx1201BinaryDecoder decoder;
  std::array<Gfx1201Wave32Phase0NextRiskEncodingStatus, kNextRiskEncodings.size()>
      statuses{};

  for (std::size_t i = 0; i < kNextRiskEncodings.size(); ++i) {
    const Gfx1201DecoderSeedEncoding* seed =
        FindGfx1201Phase0ComputeDecoderSeed(kNextRiskEncodings[i]);
    if (seed == nullptr) {
      continue;
    }
    const ExecutableInstructionSummary executable_summary =
        SummarizeExecutableInstructions(*seed, decoder);

    statuses[i] = Gfx1201Wave32Phase0NextRiskEncodingStatus{
        seed->encoding_name,
        seed->example_instruction,
        seed->rationale,
        seed->instruction_count,
        executable_summary.executable_instruction_count,
        executable_summary.first_executable_instruction,
        seed->transferable_as_is_count,
        seed->transferable_with_decoder_work_count,
        seed->transferable_with_semantic_work_count,
        seed->transferable_with_decoder_and_semantic_work_count,
        seed->gfx1201_specific_count,
    };
  }

  return statuses;
}

std::vector<Gfx1201Wave32Phase0VdsBoundaryInstructionStatus>
BuildRemainingVdsInstructionStatuses() {
  std::vector<Gfx1201Wave32Phase0VdsBoundaryInstructionStatus> statuses;
  std::uint32_t tail_ordinal = 0;
  for (const Gfx1201Wave32Phase0VdsBoundaryBucket& bucket :
       kVdsBoundaryBuckets) {
    std::uint32_t bucket_ordinal = 0;
    for (std::string_view instruction_name : bucket.instruction_names) {
      const Gfx1201DecoderSeedEntry* seed_entry = FindVdsSeedEntry(instruction_name);
      statuses.push_back(Gfx1201Wave32Phase0VdsBoundaryInstructionStatus{
          instruction_name,
          bucket.bucket_name,
          bucket.blocking_dimension,
          bucket.risk_rank,
          tail_ordinal,
          bucket_ordinal,
          seed_entry == nullptr ? 0u : seed_entry->opcode,
          static_cast<std::uint16_t>(seed_entry == nullptr ? 0u
                                                           : seed_entry->operand_count),
          seed_entry == nullptr ? std::string_view{} : ToString(seed_entry->rollup),
          seed_entry == nullptr ? std::string_view{} : ToString(seed_entry->state),
          bucket.safe_under_current_request,
      });
      ++tail_ordinal;
      ++bucket_ordinal;
    }
  }
  return statuses;
}

std::vector<Gfx1201Wave32Phase0VdsBoundaryBucketStatus>
BuildVdsBoundaryBucketStatuses() {
  std::vector<Gfx1201Wave32Phase0VdsBoundaryBucketStatus> statuses;
  statuses.reserve(kVdsBoundaryBuckets.size());

  for (const Gfx1201Wave32Phase0VdsBoundaryBucket& bucket : kVdsBoundaryBuckets) {
    const std::vector<BucketOpcodePoint> opcode_points =
        CollectBucketOpcodePoints(bucket);
    Gfx1201Wave32Phase0VdsBoundaryBucketStatus status{
        bucket.bucket_name,
        bucket.blocking_dimension,
        bucket.risk_rank,
        bucket.instruction_count,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        0u,
        bucket.safe_under_current_request,
    };
    bool saw_seed_entry = false;
    std::uint32_t previous_opcode = 0;

    for (const BucketOpcodePoint& point : opcode_points) {
      const Gfx1201DecoderSeedEntry* seed_entry =
          FindVdsSeedEntry(point.instruction_name);
      if (seed_entry == nullptr) {
        continue;
      }

      if (!saw_seed_entry) {
        status.first_opcode = seed_entry->opcode;
        status.last_opcode = seed_entry->opcode;
        status.opcode_segment_count = 1u;
        status.min_operand_count =
            static_cast<std::uint16_t>(seed_entry->operand_count);
        status.max_operand_count =
            static_cast<std::uint16_t>(seed_entry->operand_count);
        saw_seed_entry = true;
      } else {
        if (seed_entry->opcode != previous_opcode + 1u) {
          ++status.opcode_segment_count;
          status.largest_opcode_gap =
              std::max(status.largest_opcode_gap,
                       seed_entry->opcode - previous_opcode - 1u);
        }
        status.first_opcode = std::min(status.first_opcode, seed_entry->opcode);
        status.last_opcode = std::max(status.last_opcode, seed_entry->opcode);
        status.min_operand_count = std::min(
            status.min_operand_count,
            static_cast<std::uint16_t>(seed_entry->operand_count));
        status.max_operand_count = std::max(
            status.max_operand_count,
            static_cast<std::uint16_t>(seed_entry->operand_count));
      }
      previous_opcode = seed_entry->opcode;

      switch (seed_entry->operand_count) {
        case 3:
          ++status.operand_count_3_count;
          break;
        case 4:
          ++status.operand_count_4_count;
          break;
        case 5:
          ++status.operand_count_5_count;
          break;
        case 6:
          ++status.operand_count_6_count;
          break;
        default:
          break;
      }

      switch (seed_entry->rollup) {
        case Gfx1201SupportRollup::kTransferableAsIs:
          ++status.transferable_as_is_rollup_count;
          break;
        case Gfx1201SupportRollup::kTransferableWithDecoderWork:
          ++status.transferable_with_decoder_work_rollup_count;
          break;
        case Gfx1201SupportRollup::kTransferableWithSemanticWork:
          ++status.transferable_with_semantic_work_rollup_count;
          break;
        case Gfx1201SupportRollup::kGfx1201Specific:
          ++status.gfx1201_specific_rollup_count;
          break;
        default:
          break;
      }

      switch (seed_entry->state) {
        case Gfx1201SupportState::kTransferableAsIs:
          ++status.transferable_as_is_state_count;
          break;
        case Gfx1201SupportState::kTransferableWithDecoderWork:
          ++status.transferable_with_decoder_work_state_count;
          break;
        case Gfx1201SupportState::kTransferableWithSemanticWork:
          ++status.transferable_with_semantic_work_state_count;
          break;
        case Gfx1201SupportState::kTransferableWithDecoderAndSemanticWork:
          ++status.transferable_with_decoder_and_semantic_work_state_count;
          break;
        case Gfx1201SupportState::kGfx1201Specific:
          ++status.gfx1201_specific_state_count;
          break;
      }
    }

    if (!opcode_points.empty()) {
      status.opcode_span_width = status.last_opcode - status.first_opcode + 1u;
      status.opcode_hole_count =
          status.opcode_span_width > status.instruction_count
              ? status.opcode_span_width - status.instruction_count
              : 0u;

      std::uint32_t current_segment_instruction_count = 1u;
      for (std::size_t i = 1; i < opcode_points.size(); ++i) {
        if (opcode_points[i].opcode == opcode_points[i - 1].opcode + 1u) {
          ++current_segment_instruction_count;
        } else {
          if (current_segment_instruction_count == 1u) {
            ++status.singleton_opcode_segment_count;
          } else {
            ++status.multi_instruction_opcode_segment_count;
          }
          status.longest_opcode_segment_instruction_count =
              std::max(status.longest_opcode_segment_instruction_count,
                       current_segment_instruction_count);
          current_segment_instruction_count = 1u;
        }
      }
      if (current_segment_instruction_count == 1u) {
        ++status.singleton_opcode_segment_count;
      } else {
        ++status.multi_instruction_opcode_segment_count;
      }
      status.longest_opcode_segment_instruction_count =
          std::max(status.longest_opcode_segment_instruction_count,
                   current_segment_instruction_count);
    }

    statuses.push_back(status);
  }

  return statuses;
}

std::vector<Gfx1201Wave32Phase0VdsOpcodeSegment> BuildVdsOpcodeSegments() {
  std::vector<Gfx1201Wave32Phase0VdsOpcodeSegment> segments;

  for (const Gfx1201Wave32Phase0VdsBoundaryBucket& bucket : kVdsBoundaryBuckets) {
    const std::vector<BucketOpcodePoint> opcode_points =
        CollectBucketOpcodePoints(bucket);
    if (opcode_points.empty()) {
      continue;
    }

    std::uint32_t segment_ordinal = 0;
    std::uint32_t first_opcode = opcode_points.front().opcode;
    std::uint32_t last_opcode = opcode_points.front().opcode;
    std::uint32_t instruction_count = 1u;
    std::string_view first_instruction_name = opcode_points.front().instruction_name;
    std::string_view last_instruction_name = opcode_points.front().instruction_name;

    for (std::size_t i = 1; i < opcode_points.size(); ++i) {
      if (opcode_points[i].opcode == last_opcode + 1u) {
        last_opcode = opcode_points[i].opcode;
        last_instruction_name = opcode_points[i].instruction_name;
        ++instruction_count;
        continue;
      }

      segments.push_back(Gfx1201Wave32Phase0VdsOpcodeSegment{
          bucket.bucket_name,
          segment_ordinal,
          first_opcode,
          last_opcode,
          instruction_count,
          first_instruction_name,
          last_instruction_name,
      });
      ++segment_ordinal;

      first_opcode = opcode_points[i].opcode;
      last_opcode = opcode_points[i].opcode;
      instruction_count = 1u;
      first_instruction_name = opcode_points[i].instruction_name;
      last_instruction_name = opcode_points[i].instruction_name;
    }

    segments.push_back(Gfx1201Wave32Phase0VdsOpcodeSegment{
        bucket.bucket_name,
        segment_ordinal,
        first_opcode,
        last_opcode,
        instruction_count,
        first_instruction_name,
        last_instruction_name,
    });
  }

  return segments;
}

std::vector<Gfx1201Wave32Phase0VdsOpcodeGap> BuildVdsOpcodeGaps() {
  std::vector<Gfx1201Wave32Phase0VdsOpcodeGap> gaps;

  for (const Gfx1201Wave32Phase0VdsBoundaryBucket& bucket : kVdsBoundaryBuckets) {
    const std::vector<BucketOpcodePoint> opcode_points =
        CollectBucketOpcodePoints(bucket);
    if (opcode_points.size() < 2u) {
      continue;
    }

    std::uint32_t segment_ordinal = 0u;
    std::uint32_t previous_segment_ordinal = 0u;
    std::uint32_t previous_opcode = opcode_points.front().opcode;
    std::string_view previous_instruction_name =
        opcode_points.front().instruction_name;

    for (std::size_t i = 1; i < opcode_points.size(); ++i) {
      const BucketOpcodePoint& point = opcode_points[i];
      if (point.opcode == previous_opcode + 1u) {
        previous_opcode = point.opcode;
        previous_instruction_name = point.instruction_name;
        continue;
      }

      const std::uint32_t next_segment_ordinal = segment_ordinal + 1u;
      gaps.push_back(Gfx1201Wave32Phase0VdsOpcodeGap{
          bucket.bucket_name,
          0u,
          previous_segment_ordinal,
          next_segment_ordinal,
          previous_opcode,
          point.opcode,
          point.opcode - previous_opcode - 1u,
          previous_instruction_name,
          point.instruction_name,
      });

      segment_ordinal = next_segment_ordinal;
      previous_segment_ordinal = segment_ordinal;
      previous_opcode = point.opcode;
      previous_instruction_name = point.instruction_name;
    }
  }

  std::uint32_t gap_ordinal = 0u;
  std::string_view current_bucket_name;
  for (Gfx1201Wave32Phase0VdsOpcodeGap& gap : gaps) {
    if (gap.bucket_name != current_bucket_name) {
      current_bucket_name = gap.bucket_name;
      gap_ordinal = 0u;
    }
    gap.gap_ordinal = gap_ordinal++;
  }

  return gaps;
}

std::vector<Gfx1201Wave32Phase0VdsNextRiskStep> BuildVdsNextRiskSteps() {
  std::vector<Gfx1201Wave32Phase0VdsNextRiskStep> steps;
  steps.reserve(kVdsBoundaryBuckets.size());

  std::uint32_t total_remaining_instruction_count = 0;
  for (const Gfx1201Wave32Phase0VdsBoundaryBucket& bucket : kVdsBoundaryBuckets) {
    total_remaining_instruction_count += bucket.instruction_count;
  }

  for (std::size_t i = 0; i < kVdsBoundaryBuckets.size(); ++i) {
    const Gfx1201Wave32Phase0VdsBoundaryBucket& bucket = kVdsBoundaryBuckets[i];
    const bool has_next_bucket = i + 1 < kVdsBoundaryBuckets.size();
    const Gfx1201Wave32Phase0VdsBoundaryBucket* next_bucket =
        has_next_bucket ? &kVdsBoundaryBuckets[i + 1] : nullptr;

    const std::uint32_t consumed_instruction_count =
        bucket.ending_instruction_ordinal + 1u;
    const std::uint32_t remaining_instruction_count_after_bucket =
        total_remaining_instruction_count > consumed_instruction_count
            ? total_remaining_instruction_count - consumed_instruction_count
            : 0u;

    steps.push_back(Gfx1201Wave32Phase0VdsNextRiskStep{
        bucket.bucket_name,
        bucket.blocking_dimension,
        bucket.risk_rank,
        bucket.instruction_names.empty() ? std::string_view{}
                                         : bucket.instruction_names.front(),
        bucket.instruction_names.empty() ? std::string_view{}
                                         : bucket.instruction_names.back(),
        bucket.instruction_count,
        total_remaining_instruction_count - bucket.starting_instruction_ordinal,
        remaining_instruction_count_after_bucket,
        next_bucket == nullptr ? std::string_view{} : next_bucket->bucket_name,
        next_bucket == nullptr ? std::string_view{}
                               : next_bucket->blocking_dimension,
        next_bucket == nullptr || next_bucket->instruction_names.empty()
            ? std::string_view{}
            : next_bucket->instruction_names.front(),
    });
  }

  return steps;
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

std::span<const Gfx1201Wave32Phase0VdsBoundaryBucket>
GetGfx1201Wave32Phase0VdsBoundaryBuckets() {
  return kVdsBoundaryBuckets;
}

std::span<const Gfx1201Wave32Phase0VdsBoundaryBucketStatus>
GetGfx1201Wave32Phase0VdsBoundaryBucketStatuses() {
  static const std::vector<Gfx1201Wave32Phase0VdsBoundaryBucketStatus>
      kStatuses = BuildVdsBoundaryBucketStatuses();
  return kStatuses;
}

std::span<const std::string_view> GetGfx1201Wave32Phase0VdsBoundaryOrder() {
  return kVdsBoundaryOrder;
}

std::span<const Gfx1201Wave32Phase0VdsBoundaryInstructionStatus>
GetGfx1201Wave32Phase0RemainingVdsInstructionStatuses() {
  static const std::vector<Gfx1201Wave32Phase0VdsBoundaryInstructionStatus>
      kStatuses = BuildRemainingVdsInstructionStatuses();
  return kStatuses;
}

std::span<const Gfx1201Wave32Phase0VdsOpcodeSegment>
GetGfx1201Wave32Phase0VdsOpcodeSegments() {
  static const std::vector<Gfx1201Wave32Phase0VdsOpcodeSegment> kSegments =
      BuildVdsOpcodeSegments();
  return kSegments;
}

std::span<const Gfx1201Wave32Phase0VdsOpcodeGap>
GetGfx1201Wave32Phase0VdsOpcodeGaps() {
  static const std::vector<Gfx1201Wave32Phase0VdsOpcodeGap> kGaps =
      BuildVdsOpcodeGaps();
  return kGaps;
}

std::span<const Gfx1201Wave32Phase0VdsNextRiskStep>
GetGfx1201Wave32Phase0VdsNextRiskSteps() {
  static const std::vector<Gfx1201Wave32Phase0VdsNextRiskStep> kSteps =
      BuildVdsNextRiskSteps();
  return kSteps;
}

const Gfx1201Wave32Phase0VdsBoundaryBucket*
FindGfx1201Wave32Phase0VdsBoundaryBucket(std::string_view bucket_name) {
  for (const Gfx1201Wave32Phase0VdsBoundaryBucket& bucket :
       GetGfx1201Wave32Phase0VdsBoundaryBuckets()) {
    if (bucket.bucket_name == bucket_name) {
      return &bucket;
    }
  }
  return nullptr;
}

const Gfx1201Wave32Phase0VdsBoundaryBucketStatus*
FindGfx1201Wave32Phase0VdsBoundaryBucketStatus(std::string_view bucket_name) {
  for (const Gfx1201Wave32Phase0VdsBoundaryBucketStatus& status :
       GetGfx1201Wave32Phase0VdsBoundaryBucketStatuses()) {
    if (status.bucket_name == bucket_name) {
      return &status;
    }
  }
  return nullptr;
}

const Gfx1201Wave32Phase0VdsBoundaryBucket*
FindGfx1201Wave32Phase0VdsBoundaryBucketForInstruction(
    std::string_view instruction_name) {
  for (const Gfx1201Wave32Phase0VdsBoundaryBucket& bucket :
       GetGfx1201Wave32Phase0VdsBoundaryBuckets()) {
    for (std::string_view candidate : bucket.instruction_names) {
      if (candidate == instruction_name) {
        return &bucket;
      }
    }
  }
  return nullptr;
}

const Gfx1201Wave32Phase0VdsBoundaryInstructionStatus*
FindGfx1201Wave32Phase0RemainingVdsInstructionStatus(
    std::string_view instruction_name) {
  for (const Gfx1201Wave32Phase0VdsBoundaryInstructionStatus& status :
       GetGfx1201Wave32Phase0RemainingVdsInstructionStatuses()) {
    if (status.instruction_name == instruction_name) {
      return &status;
    }
  }
  return nullptr;
}

const Gfx1201Wave32Phase0VdsBoundaryInstructionStatus*
FindGfx1201Wave32Phase0RemainingVdsInstructionStatusByOpcode(
    std::uint32_t opcode) {
  for (const Gfx1201Wave32Phase0VdsBoundaryInstructionStatus& status :
       GetGfx1201Wave32Phase0RemainingVdsInstructionStatuses()) {
    if (status.opcode == opcode) {
      return &status;
    }
  }
  return nullptr;
}

const Gfx1201Wave32Phase0VdsOpcodeSegment*
FindGfx1201Wave32Phase0VdsOpcodeSegment(std::string_view bucket_name,
                                        std::uint32_t segment_ordinal) {
  for (const Gfx1201Wave32Phase0VdsOpcodeSegment& segment :
       GetGfx1201Wave32Phase0VdsOpcodeSegments()) {
    if (segment.bucket_name == bucket_name &&
        segment.segment_ordinal == segment_ordinal) {
      return &segment;
    }
  }
  return nullptr;
}

const Gfx1201Wave32Phase0VdsOpcodeGap* FindGfx1201Wave32Phase0VdsOpcodeGap(
    std::string_view bucket_name, std::uint32_t gap_ordinal) {
  for (const Gfx1201Wave32Phase0VdsOpcodeGap& gap :
       GetGfx1201Wave32Phase0VdsOpcodeGaps()) {
    if (gap.bucket_name == bucket_name && gap.gap_ordinal == gap_ordinal) {
      return &gap;
    }
  }
  return nullptr;
}

const Gfx1201Wave32Phase0VdsNextRiskStep*
FindGfx1201Wave32Phase0VdsNextRiskStep(std::string_view bucket_name) {
  for (const Gfx1201Wave32Phase0VdsNextRiskStep& step :
       GetGfx1201Wave32Phase0VdsNextRiskSteps()) {
    if (step.bucket_name == bucket_name) {
      return &step;
    }
  }
  return nullptr;
}

bool HasGfx1201Wave32SafeVdsContinuation() {
  for (const Gfx1201Wave32Phase0VdsBoundaryBucket& bucket :
       GetGfx1201Wave32Phase0VdsBoundaryBuckets()) {
    if (bucket.safe_under_current_request) {
      return true;
    }
  }
  return false;
}

std::string_view GetGfx1201Wave32RecommendedNextVdsBucket() {
  for (const Gfx1201Wave32Phase0VdsBoundaryBucket& bucket :
       GetGfx1201Wave32Phase0VdsBoundaryBuckets()) {
    if (bucket.safe_under_current_request) {
      return bucket.bucket_name;
    }
  }
  return {};
}

std::string_view GetGfx1201Wave32FirstUnsafeVdsBucket() {
  for (const Gfx1201Wave32Phase0VdsBoundaryBucket& bucket :
       GetGfx1201Wave32Phase0VdsBoundaryBuckets()) {
    if (!bucket.safe_under_current_request) {
      return bucket.bucket_name;
    }
  }
  return {};
}

std::string_view GetGfx1201Wave32FirstUnsafeVdsBlockingDimension() {
  for (const Gfx1201Wave32Phase0VdsBoundaryBucket& bucket :
       GetGfx1201Wave32Phase0VdsBoundaryBuckets()) {
    if (!bucket.safe_under_current_request) {
      return bucket.blocking_dimension;
    }
  }
  return {};
}

std::span<const std::string_view> GetGfx1201Wave32FirstUnsafeVdsInstructions() {
  for (const Gfx1201Wave32Phase0VdsBoundaryBucket& bucket :
       GetGfx1201Wave32Phase0VdsBoundaryBuckets()) {
    if (!bucket.safe_under_current_request) {
      return bucket.instruction_names;
    }
  }
  return {};
}

std::string_view GetGfx1201Wave32Phase0RecommendedNextEncoding() {
  for (std::string_view encoding_name : kFrontierOrder) {
    if (!IsGfx1201Wave32Phase0EncodingSaturated(encoding_name)) {
      return encoding_name;
    }
  }
  return {};
}

bool IsGfx1201Wave32Phase0EncodingSaturated(std::string_view encoding_name) {
  const Gfx1201Wave32Phase0EncodingStatus* status =
      FindGfx1201Wave32Phase0EncodingStatus(encoding_name);
  return status != nullptr && status->fully_executable;
}

}  // namespace mirage::sim::isa
