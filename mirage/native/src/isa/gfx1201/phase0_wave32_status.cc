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
            2,
            false,
            std::span<const std::string_view>(kAppendConsumeVdsInstructions),
        },
        {
            "exchange_compare_store",
            "DS_CONDXCHG32_RTN_B64",
            "Exchange and compare-store forms are the next semantic-risk step on the VDS path.",
            "exchange_compare_store_semantics",
            7,
            false,
            std::span<const std::string_view>(kExchangeCompareStoreVdsInstructions),
        },
        {
            "multi_address",
            "DS_LOAD_2ADDR_B32",
            "Remaining two-address and stride64 LDS forms would widen the current one-address execution model.",
            "multi_address_semantics",
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
            false,
            std::span<const std::string_view>(kBvhStackVdsInstructions),
        },
    }};

struct ExecutableInstructionSummary {
  std::uint32_t executable_instruction_count = 0;
  std::string_view first_executable_instruction;
};

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

std::span<const std::string_view> GetGfx1201Wave32Phase0VdsBoundaryOrder() {
  return kVdsBoundaryOrder;
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
