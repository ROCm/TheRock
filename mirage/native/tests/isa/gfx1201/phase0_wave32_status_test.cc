#include <array>
#include <iostream>

#include "lib/sim/isa/gfx1201/binary_decoder.h"
#include "lib/sim/isa/gfx1201/phase0_wave32_status.h"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main() {
  using namespace mirage::sim::isa;

  Gfx1201BinaryDecoder decoder;
  const auto statuses = GetGfx1201Wave32Phase0EncodingStatuses();
  const Gfx1201Wave32Phase0EncodingStatus* vop1 =
      FindGfx1201Wave32Phase0EncodingStatus("ENC_VOP1");
  const Gfx1201Wave32Phase0EncodingStatus* vop2 =
      FindGfx1201Wave32Phase0EncodingStatus("ENC_VOP2");
  const Gfx1201Wave32Phase0EncodingStatus* vopc =
      FindGfx1201Wave32Phase0EncodingStatus("ENC_VOPC");
  const Gfx1201Wave32Phase0EncodingStatus* smem_status =
      FindGfx1201Wave32Phase0EncodingStatus("ENC_SMEM");
  const Gfx1201Wave32Phase0EncodingStatus* vglobal_status =
      FindGfx1201Wave32Phase0EncodingStatus("ENC_VGLOBAL");
  const Gfx1201Wave32Phase0EncodingStatus* missing =
      FindGfx1201Wave32Phase0EncodingStatus("ENC_VIMAGE");
  const auto next_risk_statuses = GetGfx1201Wave32Phase0NextRiskEncodingStatuses();
  const Gfx1201Wave32Phase0NextRiskEncodingStatus* smem =
      FindGfx1201Wave32Phase0NextRiskEncodingStatus("ENC_SMEM");
  const Gfx1201Wave32Phase0NextRiskEncodingStatus* vop3 =
      FindGfx1201Wave32Phase0NextRiskEncodingStatus("ENC_VOP3");
  const Gfx1201Wave32Phase0NextRiskEncodingStatus* vds =
      FindGfx1201Wave32Phase0NextRiskEncodingStatus("ENC_VDS");
  const Gfx1201Wave32Phase0NextRiskEncodingStatus* vglobal =
      FindGfx1201Wave32Phase0NextRiskEncodingStatus("ENC_VGLOBAL");
  const Gfx1201Wave32Phase0VdsBoundaryBucket* append_consume_bucket =
      FindGfx1201Wave32Phase0VdsBoundaryBucket("append_consume");
  const Gfx1201Wave32Phase0VdsBoundaryBucket* exchange_compare_store_bucket =
      FindGfx1201Wave32Phase0VdsBoundaryBucket("exchange_compare_store");
  const Gfx1201Wave32Phase0VdsBoundaryBucket* multi_address_bucket =
      FindGfx1201Wave32Phase0VdsBoundaryBucket("multi_address");
  const Gfx1201Wave32Phase0VdsBoundaryBucket* bvh_stack_bucket =
      FindGfx1201Wave32Phase0VdsBoundaryBucket("bvh_stack");
  const Gfx1201Wave32Phase0VdsBoundaryBucket* missing_vds_bucket =
      FindGfx1201Wave32Phase0VdsBoundaryBucket("not_a_bucket");

  if (!Expect(statuses.size() == 5u,
              "expected five tracked wave32-local encodings") ||
      !Expect(vop1 != nullptr, "expected ENC_VOP1 status") ||
      !Expect(vop2 != nullptr, "expected ENC_VOP2 status") ||
      !Expect(vopc != nullptr, "expected ENC_VOPC status") ||
      !Expect(smem_status != nullptr, "expected ENC_SMEM status") ||
      !Expect(vglobal_status != nullptr, "expected ENC_VGLOBAL status") ||
      !Expect(missing == nullptr, "expected no status for ENC_VIMAGE") ||
      !Expect(next_risk_statuses.size() == 4u,
              "expected four next-risk encoding statuses") ||
      !Expect(smem != nullptr, "expected ENC_SMEM next-risk status") ||
      !Expect(vop3 != nullptr, "expected ENC_VOP3 next-risk status") ||
      !Expect(vds != nullptr, "expected ENC_VDS next-risk status") ||
      !Expect(vglobal != nullptr, "expected ENC_VGLOBAL next-risk status") ||
      !Expect(append_consume_bucket != nullptr,
              "expected append/consume VDS bucket lookup") ||
      !Expect(exchange_compare_store_bucket != nullptr,
              "expected exchange/compare-store VDS bucket lookup") ||
      !Expect(multi_address_bucket != nullptr,
              "expected multi-address VDS bucket lookup") ||
      !Expect(bvh_stack_bucket != nullptr,
              "expected BVH-stack VDS bucket lookup") ||
      !Expect(missing_vds_bucket == nullptr,
              "expected missing VDS bucket lookup to fail")) {
    return 1;
  }

  if (!Expect(vop1->seeded_instruction_count == 90u,
              "expected ENC_VOP1 seeded instruction count") ||
      !Expect(vop1->executable_instruction_count == 90u,
              "expected ENC_VOP1 executable instruction count") ||
      !Expect(vop1->fully_executable, "expected ENC_VOP1 saturation") ||
      !Expect(vop2->seeded_instruction_count == 47u,
              "expected ENC_VOP2 seeded instruction count") ||
      !Expect(vop2->executable_instruction_count == 47u,
              "expected ENC_VOP2 executable instruction count") ||
      !Expect(vop2->fully_executable, "expected ENC_VOP2 saturation") ||
      !Expect(vopc->seeded_instruction_count == 162u,
              "expected ENC_VOPC seeded instruction count") ||
      !Expect(vopc->executable_instruction_count == 162u,
              "expected ENC_VOPC executable instruction count") ||
      !Expect(vopc->fully_executable, "expected ENC_VOPC saturation") ||
      !Expect(smem_status->seeded_instruction_count == 28u,
              "expected ENC_SMEM seeded instruction count") ||
      !Expect(smem_status->executable_instruction_count == 28u,
              "expected ENC_SMEM executable instruction count") ||
      !Expect(smem_status->fully_executable,
              "expected ENC_SMEM saturation") ||
      !Expect(vglobal_status->seeded_instruction_count == 65u,
              "expected ENC_VGLOBAL seeded instruction count") ||
      !Expect(vglobal_status->executable_instruction_count == 65u,
              "expected ENC_VGLOBAL executable instruction count") ||
      !Expect(vglobal_status->fully_executable,
              "expected ENC_VGLOBAL saturation")) {
    return 1;
  }

  if (!Expect(smem->seeded_instruction_count == 28u,
              "expected ENC_SMEM seeded instruction count") ||
      !Expect(smem->executable_instruction_count == 28u,
              "expected ENC_SMEM executable foothold count") ||
      !Expect(smem->HasExecutableFoothold(),
              "expected ENC_SMEM executable foothold helper") ||
      !Expect(smem->TransferableWithDecoderRollupCount() == 3u,
              "expected ENC_SMEM decoder rollup count") ||
      !Expect(smem->gfx1201_specific_count == 25u,
              "expected ENC_SMEM gfx1201-specific count") ||
      !Expect(vop3->seeded_instruction_count == 434u,
              "expected ENC_VOP3 seeded instruction count") ||
      !Expect(vop3->TransferableWithDecoderRollupCount() == 91u,
              "expected ENC_VOP3 decoder rollup count") ||
      !Expect(vop3->transferable_with_semantic_work_count == 24u,
              "expected ENC_VOP3 semantic-work count") ||
      !Expect(vop3->gfx1201_specific_count == 87u,
              "expected ENC_VOP3 gfx1201-specific count") ||
      !Expect(vds->seeded_instruction_count == 123u,
              "expected ENC_VDS seeded instruction count") ||
      !Expect(vds->executable_instruction_count == 99u,
              "expected ENC_VDS executable foothold count") ||
      !Expect(vds->HasExecutableFoothold(),
              "expected ENC_VDS executable foothold helper") ||
      !Expect(vds->TransferableWithDecoderRollupCount() == 38u,
              "expected ENC_VDS decoder rollup count") ||
      !Expect(vds->gfx1201_specific_count == 58u,
              "expected ENC_VDS gfx1201-specific count") ||
      !Expect(vglobal->seeded_instruction_count == 65u,
              "expected ENC_VGLOBAL seeded instruction count") ||
      !Expect(vglobal->executable_instruction_count == 65u,
              "expected ENC_VGLOBAL executable foothold count") ||
      !Expect(vglobal->HasExecutableFoothold(),
              "expected ENC_VGLOBAL executable foothold helper") ||
      !Expect(vglobal->transferable_as_is_count == 3u,
              "expected ENC_VGLOBAL as-is count") ||
      !Expect(vglobal->gfx1201_specific_count == 62u,
              "expected ENC_VGLOBAL gfx1201-specific count")) {
    return 1;
  }

  if (!Expect(decoder.Phase0ExecutableOpcodes().size() == 517u,
              "expected phase-0 executable opcode count") ||
      !Expect(IsGfx1201Wave32Phase0EncodingSaturated("ENC_VOP1"),
              "expected ENC_VOP1 saturation helper") ||
      !Expect(IsGfx1201Wave32Phase0EncodingSaturated("ENC_VOP2"),
              "expected ENC_VOP2 saturation helper") ||
      !Expect(IsGfx1201Wave32Phase0EncodingSaturated("ENC_VOPC"),
              "expected ENC_VOPC saturation helper") ||
      !Expect(IsGfx1201Wave32Phase0EncodingSaturated("ENC_SMEM"),
              "expected ENC_SMEM saturation helper") ||
      !Expect(IsGfx1201Wave32Phase0EncodingSaturated("ENC_VGLOBAL"),
              "expected ENC_VGLOBAL saturation helper") ||
      !Expect(!IsGfx1201Wave32Phase0EncodingSaturated("ENC_VOP3"),
              "expected ENC_VOP3 to be out of scope")) {
    return 1;
  }

  constexpr std::array<const char*, 4> kExpectedNextRiskEncodings{{
      "ENC_SMEM",
      "ENC_VOP3",
      "ENC_VDS",
      "ENC_VGLOBAL",
  }};
  constexpr std::array<const char*, 4> kExpectedFrontierOrder{{
      "ENC_SMEM",
      "ENC_VGLOBAL",
      "ENC_VDS",
      "ENC_VOP3",
  }};
  constexpr std::array<const char*, 4> kExpectedVdsBoundaryOrder{{
      "append_consume",
      "exchange_compare_store",
      "multi_address",
      "bvh_stack",
  }};
  constexpr std::array<const char*, 2> kExpectedFirstUnsafeVdsInstructions{{
      "DS_APPEND",
      "DS_CONSUME",
  }};

  const auto next_risk_encodings = GetGfx1201Wave32Phase0NextRiskEncodings();
  const auto frontier_order = GetGfx1201Wave32Phase0FrontierOrder();
  const auto vds_boundary_buckets = GetGfx1201Wave32Phase0VdsBoundaryBuckets();
  const auto vds_boundary_bucket_statuses =
      GetGfx1201Wave32Phase0VdsBoundaryBucketStatuses();
  const auto vds_boundary_order = GetGfx1201Wave32Phase0VdsBoundaryOrder();
  const auto remaining_vds_instruction_statuses =
      GetGfx1201Wave32Phase0RemainingVdsInstructionStatuses();
  const auto vds_next_risk_steps = GetGfx1201Wave32Phase0VdsNextRiskSteps();
  const auto first_unsafe_vds_instructions =
      GetGfx1201Wave32FirstUnsafeVdsInstructions();
  const Gfx1201Wave32Phase0VdsBoundaryBucketStatus* append_bucket_status =
      FindGfx1201Wave32Phase0VdsBoundaryBucketStatus("append_consume");
  const Gfx1201Wave32Phase0VdsBoundaryBucketStatus* exchange_bucket_status =
      FindGfx1201Wave32Phase0VdsBoundaryBucketStatus("exchange_compare_store");
  const Gfx1201Wave32Phase0VdsBoundaryBucketStatus* multi_bucket_status =
      FindGfx1201Wave32Phase0VdsBoundaryBucketStatus("multi_address");
  const Gfx1201Wave32Phase0VdsBoundaryBucketStatus* bvh_bucket_status =
      FindGfx1201Wave32Phase0VdsBoundaryBucketStatus("bvh_stack");
  const Gfx1201Wave32Phase0VdsBoundaryBucketStatus* missing_bucket_status =
      FindGfx1201Wave32Phase0VdsBoundaryBucketStatus("not_a_bucket");
  const Gfx1201Wave32Phase0VdsBoundaryInstructionStatus* append_status =
      FindGfx1201Wave32Phase0RemainingVdsInstructionStatus("DS_APPEND");
  const Gfx1201Wave32Phase0VdsBoundaryInstructionStatus* consume_opcode_status =
      FindGfx1201Wave32Phase0RemainingVdsInstructionStatusByOpcode(61u);
  const Gfx1201Wave32Phase0VdsBoundaryInstructionStatus*
      condxchg_opcode_status =
          FindGfx1201Wave32Phase0RemainingVdsInstructionStatusByOpcode(126u);
  const Gfx1201Wave32Phase0VdsBoundaryInstructionStatus* storexchg_status =
      FindGfx1201Wave32Phase0RemainingVdsInstructionStatus(
          "DS_STOREXCHG_RTN_B64");
  const Gfx1201Wave32Phase0VdsBoundaryInstructionStatus* stride_status =
      FindGfx1201Wave32Phase0RemainingVdsInstructionStatus(
          "DS_LOAD_2ADDR_STRIDE64_B64");
  const Gfx1201Wave32Phase0VdsBoundaryInstructionStatus*
      stride_opcode_status =
          FindGfx1201Wave32Phase0RemainingVdsInstructionStatusByOpcode(120u);
  const Gfx1201Wave32Phase0VdsBoundaryInstructionStatus* bvh_status =
      FindGfx1201Wave32Phase0RemainingVdsInstructionStatus(
          "DS_BVH_STACK_PUSH8_POP2_RTN_B64");
  const Gfx1201Wave32Phase0VdsBoundaryInstructionStatus* bvh_opcode_status =
      FindGfx1201Wave32Phase0RemainingVdsInstructionStatusByOpcode(226u);
  const Gfx1201Wave32Phase0VdsBoundaryInstructionStatus* missing_vds_status =
      FindGfx1201Wave32Phase0RemainingVdsInstructionStatus(
          "DS_BPERMUTE_FI_B32");
  const Gfx1201Wave32Phase0VdsBoundaryInstructionStatus*
      missing_vds_opcode_status =
          FindGfx1201Wave32Phase0RemainingVdsInstructionStatusByOpcode(53u);
  const Gfx1201Wave32Phase0VdsNextRiskStep* append_step =
      FindGfx1201Wave32Phase0VdsNextRiskStep("append_consume");
  const Gfx1201Wave32Phase0VdsNextRiskStep* exchange_step =
      FindGfx1201Wave32Phase0VdsNextRiskStep("exchange_compare_store");
  const Gfx1201Wave32Phase0VdsNextRiskStep* multi_address_step =
      FindGfx1201Wave32Phase0VdsNextRiskStep("multi_address");
  const Gfx1201Wave32Phase0VdsNextRiskStep* bvh_step =
      FindGfx1201Wave32Phase0VdsNextRiskStep("bvh_stack");
  const Gfx1201Wave32Phase0VdsNextRiskStep* missing_vds_step =
      FindGfx1201Wave32Phase0VdsNextRiskStep("not_a_bucket");
  if (!Expect(next_risk_encodings.size() == kExpectedNextRiskEncodings.size(),
              "expected next-risk encoding count") ||
      !Expect(frontier_order.size() == kExpectedFrontierOrder.size(),
              "expected frontier order count") ||
      !Expect(vds_boundary_buckets.size() == 4u,
              "expected four VDS boundary buckets") ||
      !Expect(vds_boundary_bucket_statuses.size() == 4u,
              "expected four VDS boundary bucket statuses") ||
      !Expect(vds_boundary_order.size() == kExpectedVdsBoundaryOrder.size(),
              "expected VDS boundary order count") ||
      !Expect(remaining_vds_instruction_statuses.size() == 24u,
              "expected remaining VDS instruction status count") ||
      !Expect(vds_next_risk_steps.size() == 4u,
              "expected remaining VDS next-risk step count") ||
      !Expect(first_unsafe_vds_instructions.size() ==
                  kExpectedFirstUnsafeVdsInstructions.size(),
              "expected first unsafe VDS instruction count") ||
      !Expect(!HasGfx1201Wave32SafeVdsContinuation(),
              "expected no safe VDS continuation under the current boundary") ||
      !Expect(GetGfx1201Wave32RecommendedNextVdsBucket().empty(),
              "expected no recommended VDS bucket under the current boundary") ||
      !Expect(GetGfx1201Wave32FirstUnsafeVdsBucket() == "append_consume",
              "expected append/consume as the first unsafe VDS bucket") ||
      !Expect(GetGfx1201Wave32FirstUnsafeVdsBlockingDimension() ==
                  "allocator_or_gds_semantics",
              "expected append/consume blocking dimension as first unsafe VDS boundary") ||
      !Expect(append_bucket_status != nullptr && exchange_bucket_status != nullptr &&
                  multi_bucket_status != nullptr && bvh_bucket_status != nullptr,
              "expected VDS bucket status lookups") ||
      !Expect(missing_bucket_status == nullptr,
              "expected missing VDS bucket status lookup to fail") ||
      !Expect(append_status != nullptr && consume_opcode_status != nullptr &&
                  condxchg_opcode_status != nullptr &&
                  storexchg_status != nullptr && stride_status != nullptr &&
                  stride_opcode_status != nullptr && bvh_status != nullptr &&
                  bvh_opcode_status != nullptr,
              "expected remaining VDS instruction lookups") ||
      !Expect(append_step != nullptr && exchange_step != nullptr &&
                  multi_address_step != nullptr && bvh_step != nullptr,
              "expected remaining VDS next-risk step lookups") ||
      !Expect(missing_vds_status == nullptr,
              "expected executable VDS op to stay out of remaining-VDS status") ||
      !Expect(missing_vds_opcode_status == nullptr,
              "expected executable VDS opcode to stay out of remaining-VDS status") ||
      !Expect(missing_vds_step == nullptr,
              "expected missing VDS next-risk step lookup to fail") ||
      !Expect(GetGfx1201Wave32Phase0RecommendedNextEncoding() == "ENC_VDS",
              "expected ENC_VDS as the recommended next frontier")) {
    return 1;
  }

  for (std::size_t i = 0; i < kExpectedNextRiskEncodings.size(); ++i) {
    if (!Expect(next_risk_encodings[i] == kExpectedNextRiskEncodings[i],
                "unexpected next-risk encoding order")) {
      return 1;
    }
  }

  for (std::size_t i = 0; i < kExpectedFrontierOrder.size(); ++i) {
    if (!Expect(frontier_order[i] == kExpectedFrontierOrder[i],
                "unexpected frontier order")) {
      return 1;
    }
  }

  for (std::size_t i = 0; i < kExpectedVdsBoundaryOrder.size(); ++i) {
    if (!Expect(vds_boundary_order[i] == kExpectedVdsBoundaryOrder[i],
                "unexpected VDS boundary order")) {
      return 1;
    }
  }

  for (std::size_t i = 0; i < kExpectedFirstUnsafeVdsInstructions.size(); ++i) {
    if (!Expect(first_unsafe_vds_instructions[i] ==
                    kExpectedFirstUnsafeVdsInstructions[i],
                "unexpected first unsafe VDS instruction order")) {
      return 1;
    }
  }

  if (!Expect(vds_boundary_bucket_statuses.front().bucket_name ==
                  "append_consume" &&
                  vds_boundary_bucket_statuses.front().instruction_count == 2u &&
                  vds_boundary_bucket_statuses.front().first_opcode == 61u &&
                  vds_boundary_bucket_statuses.front().last_opcode == 62u &&
                  vds_boundary_bucket_statuses.front().min_operand_count == 3u &&
                  vds_boundary_bucket_statuses.front().max_operand_count == 3u &&
                  vds_boundary_bucket_statuses.front()
                          .transferable_with_decoder_work_count == 2u &&
                  vds_boundary_bucket_statuses.front()
                          .transferable_with_decoder_and_semantic_work_count ==
                      2u &&
                  vds_boundary_bucket_statuses.front().gfx1201_specific_count ==
                      0u,
              "expected first VDS bucket status") ||
      !Expect(vds_boundary_bucket_statuses.back().bucket_name == "bvh_stack" &&
                  vds_boundary_bucket_statuses.back().instruction_count == 3u &&
                  vds_boundary_bucket_statuses.back().first_opcode == 224u &&
                  vds_boundary_bucket_statuses.back().last_opcode == 226u &&
                  vds_boundary_bucket_statuses.back().min_operand_count == 4u &&
                  vds_boundary_bucket_statuses.back().max_operand_count == 4u &&
                  vds_boundary_bucket_statuses.back()
                          .transferable_with_decoder_work_count == 0u &&
                  vds_boundary_bucket_statuses.back()
                          .transferable_with_decoder_and_semantic_work_count ==
                      0u &&
                  vds_boundary_bucket_statuses.back().gfx1201_specific_count ==
                      3u,
              "expected last VDS bucket status") ||
      !Expect(append_bucket_status->blocking_dimension ==
                      "allocator_or_gds_semantics" &&
                  append_bucket_status->risk_rank == 0u &&
                  append_bucket_status->instruction_count == 2u &&
                  append_bucket_status->first_opcode == 61u &&
                  append_bucket_status->last_opcode == 62u &&
                  append_bucket_status->min_operand_count == 3u &&
                  append_bucket_status->max_operand_count == 3u &&
                  append_bucket_status->transferable_with_decoder_work_count ==
                      2u &&
                  append_bucket_status
                          ->transferable_with_decoder_and_semantic_work_count ==
                      2u &&
                  append_bucket_status->gfx1201_specific_count == 0u &&
                  !append_bucket_status->safe_under_current_request,
              "expected append bucket summary status") ||
      !Expect(exchange_bucket_status->blocking_dimension ==
                      "exchange_compare_store_semantics" &&
                  exchange_bucket_status->risk_rank == 1u &&
                  exchange_bucket_status->instruction_count == 7u &&
                  exchange_bucket_status->first_opcode == 16u &&
                  exchange_bucket_status->last_opcode == 126u &&
                  exchange_bucket_status->min_operand_count == 5u &&
                  exchange_bucket_status->max_operand_count == 6u &&
                  exchange_bucket_status
                          ->transferable_with_decoder_work_count == 1u &&
                  exchange_bucket_status
                          ->transferable_with_decoder_and_semantic_work_count ==
                      1u &&
                  exchange_bucket_status->gfx1201_specific_count == 6u,
              "expected exchange bucket summary status") ||
      !Expect(multi_bucket_status->blocking_dimension ==
                      "multi_address_semantics" &&
                  multi_bucket_status->risk_rank == 2u &&
                  multi_bucket_status->instruction_count == 12u &&
                  multi_bucket_status->first_opcode == 14u &&
                  multi_bucket_status->last_opcode == 120u &&
                  multi_bucket_status->min_operand_count == 3u &&
                  multi_bucket_status->max_operand_count == 6u &&
                  multi_bucket_status->transferable_with_decoder_work_count ==
                      0u &&
                  multi_bucket_status
                          ->transferable_with_decoder_and_semantic_work_count ==
                      0u &&
                  multi_bucket_status->gfx1201_specific_count == 12u,
              "expected multi-address bucket summary status") ||
      !Expect(bvh_bucket_status->blocking_dimension ==
                      "gfx1201_specific_bvh_semantics" &&
                  bvh_bucket_status->risk_rank == 3u &&
                  bvh_bucket_status->instruction_count == 3u &&
                  bvh_bucket_status->first_opcode == 224u &&
                  bvh_bucket_status->last_opcode == 226u &&
                  bvh_bucket_status->min_operand_count == 4u &&
                  bvh_bucket_status->max_operand_count == 4u &&
                  bvh_bucket_status->transferable_with_decoder_work_count ==
                      0u &&
                  bvh_bucket_status
                          ->transferable_with_decoder_and_semantic_work_count ==
                      0u &&
                  bvh_bucket_status->gfx1201_specific_count == 3u,
              "expected BVH bucket summary status")) {
    return 1;
  }

  if (!Expect(remaining_vds_instruction_statuses.front().instruction_name ==
                  "DS_APPEND" &&
                  remaining_vds_instruction_statuses.front().bucket_name ==
                      "append_consume" &&
                  remaining_vds_instruction_statuses.front().bucket_risk_rank ==
                      0u &&
                  remaining_vds_instruction_statuses.front().tail_ordinal == 0u &&
                  remaining_vds_instruction_statuses.front().bucket_ordinal ==
                      0u &&
                  remaining_vds_instruction_statuses.front().opcode == 62u &&
                  remaining_vds_instruction_statuses.front().operand_count ==
                      3u &&
                  remaining_vds_instruction_statuses.front().support_rollup ==
                      "transferable_with_decoder_work" &&
                  remaining_vds_instruction_statuses.front().support_state ==
                      "transferable_with_decoder_and_semantic_work",
              "expected first remaining VDS instruction status") ||
      !Expect(remaining_vds_instruction_statuses.back().instruction_name ==
                  "DS_BVH_STACK_PUSH8_POP2_RTN_B64" &&
                  remaining_vds_instruction_statuses.back().bucket_name ==
                      "bvh_stack" &&
                  remaining_vds_instruction_statuses.back().bucket_risk_rank ==
                      3u &&
                  remaining_vds_instruction_statuses.back().tail_ordinal ==
                      23u &&
                  remaining_vds_instruction_statuses.back().bucket_ordinal ==
                      2u &&
                  remaining_vds_instruction_statuses.back().opcode == 226u &&
                  remaining_vds_instruction_statuses.back().operand_count ==
                      4u &&
                  remaining_vds_instruction_statuses.back().support_rollup ==
                      "gfx1201_specific" &&
                  remaining_vds_instruction_statuses.back().support_state ==
                      "gfx1201_specific",
              "expected last remaining VDS instruction status") ||
      !Expect(append_status->bucket_name == "append_consume" &&
                  append_status->blocking_dimension ==
                      "allocator_or_gds_semantics" &&
                  append_status->bucket_risk_rank == 0u &&
                  append_status->tail_ordinal == 0u &&
                  append_status->bucket_ordinal == 0u &&
                  append_status->opcode == 62u &&
                  append_status->operand_count == 3u &&
                  append_status->support_rollup ==
                      "transferable_with_decoder_work" &&
                  append_status->support_state ==
                      "transferable_with_decoder_and_semantic_work" &&
                  !append_status->safe_under_current_request,
              "expected append remaining-VDS instruction status") ||
      !Expect(consume_opcode_status->instruction_name == "DS_CONSUME" &&
                  consume_opcode_status->bucket_name == "append_consume" &&
                  consume_opcode_status->tail_ordinal == 1u &&
                  consume_opcode_status->bucket_ordinal == 1u &&
                  consume_opcode_status->opcode == 61u &&
                  consume_opcode_status->operand_count == 3u &&
                  consume_opcode_status->support_rollup ==
                      "transferable_with_decoder_work" &&
                  consume_opcode_status->support_state ==
                      "transferable_with_decoder_and_semantic_work",
              "expected append/consume opcode lookup status") ||
      !Expect(condxchg_opcode_status->instruction_name ==
                      "DS_CONDXCHG32_RTN_B64" &&
                  condxchg_opcode_status->bucket_name ==
                      "exchange_compare_store" &&
                  condxchg_opcode_status->bucket_risk_rank == 1u &&
                  condxchg_opcode_status->tail_ordinal == 2u &&
                  condxchg_opcode_status->bucket_ordinal == 0u &&
                  condxchg_opcode_status->opcode == 126u &&
                  condxchg_opcode_status->operand_count == 5u &&
                  condxchg_opcode_status->support_rollup ==
                      "transferable_with_decoder_work" &&
                  condxchg_opcode_status->support_state ==
                      "transferable_with_decoder_and_semantic_work",
              "expected exchange/compare-store opcode lookup status") ||
      !Expect(storexchg_status->bucket_name == "exchange_compare_store" &&
                  storexchg_status->blocking_dimension ==
                      "exchange_compare_store_semantics" &&
                  storexchg_status->bucket_risk_rank == 1u &&
                  storexchg_status->tail_ordinal == 8u &&
                  storexchg_status->bucket_ordinal == 6u &&
                  storexchg_status->opcode == 109u &&
                  storexchg_status->operand_count == 5u &&
                  storexchg_status->support_rollup == "gfx1201_specific" &&
                  storexchg_status->support_state == "gfx1201_specific" &&
                  !storexchg_status->safe_under_current_request,
              "expected storexchg remaining-VDS instruction status") ||
      !Expect(stride_status->bucket_name == "multi_address" &&
                  stride_status->blocking_dimension ==
                      "multi_address_semantics" &&
                  stride_status->bucket_risk_rank == 2u &&
                  stride_status->tail_ordinal == 12u &&
                  stride_status->bucket_ordinal == 3u &&
                  stride_status->opcode == 120u &&
                  stride_status->operand_count == 3u &&
                  stride_status->support_rollup == "gfx1201_specific" &&
                  stride_status->support_state == "gfx1201_specific" &&
                  !stride_status->safe_under_current_request,
              "expected multi-address remaining-VDS instruction status") ||
      !Expect(stride_opcode_status->instruction_name ==
                      "DS_LOAD_2ADDR_STRIDE64_B64" &&
                  stride_opcode_status->bucket_name == "multi_address" &&
                  stride_opcode_status->tail_ordinal == 12u &&
                  stride_opcode_status->bucket_ordinal == 3u &&
                  stride_opcode_status->opcode == 120u &&
                  stride_opcode_status->operand_count == 3u &&
                  stride_opcode_status->support_rollup ==
                      "gfx1201_specific" &&
                  stride_opcode_status->support_state == "gfx1201_specific",
              "expected multi-address opcode lookup status") ||
      !Expect(bvh_status->bucket_name == "bvh_stack" &&
                  bvh_status->blocking_dimension ==
                      "gfx1201_specific_bvh_semantics" &&
                  bvh_status->bucket_risk_rank == 3u &&
                  bvh_status->tail_ordinal == 23u &&
                  bvh_status->bucket_ordinal == 2u &&
                  bvh_status->opcode == 226u &&
                  bvh_status->operand_count == 4u &&
                  bvh_status->support_rollup == "gfx1201_specific" &&
                  bvh_status->support_state == "gfx1201_specific" &&
                  !bvh_status->safe_under_current_request,
              "expected BVH remaining-VDS instruction status") ||
      !Expect(bvh_opcode_status->instruction_name ==
                      "DS_BVH_STACK_PUSH8_POP2_RTN_B64" &&
                  bvh_opcode_status->bucket_name == "bvh_stack" &&
                  bvh_opcode_status->tail_ordinal == 23u &&
                  bvh_opcode_status->bucket_ordinal == 2u &&
                  bvh_opcode_status->opcode == 226u &&
                  bvh_opcode_status->operand_count == 4u &&
                  bvh_opcode_status->support_rollup == "gfx1201_specific" &&
                  bvh_opcode_status->support_state == "gfx1201_specific",
              "expected BVH opcode lookup status")) {
    return 1;
  }

  if (!Expect(vds_next_risk_steps.front().bucket_name == "append_consume" &&
                  vds_next_risk_steps.front().first_instruction_name ==
                      "DS_APPEND" &&
                  vds_next_risk_steps.front().last_instruction_name ==
                      "DS_CONSUME" &&
                  vds_next_risk_steps.front()
                          .remaining_instruction_count_including_bucket ==
                      24u &&
                  vds_next_risk_steps.front()
                          .remaining_instruction_count_after_bucket == 22u &&
                  vds_next_risk_steps.front().next_bucket_name ==
                      "exchange_compare_store" &&
                  vds_next_risk_steps.front().next_bucket_blocking_dimension ==
                      "exchange_compare_store_semantics" &&
                  vds_next_risk_steps.front().next_instruction_name ==
                      "DS_CONDXCHG32_RTN_B64",
              "expected first VDS next-risk step") ||
      !Expect(vds_next_risk_steps.back().bucket_name == "bvh_stack" &&
                  vds_next_risk_steps.back().first_instruction_name ==
                      "DS_BVH_STACK_PUSH4_POP1_RTN_B32" &&
                  vds_next_risk_steps.back().last_instruction_name ==
                      "DS_BVH_STACK_PUSH8_POP2_RTN_B64" &&
                  vds_next_risk_steps.back()
                          .remaining_instruction_count_including_bucket ==
                      3u &&
                  vds_next_risk_steps.back()
                          .remaining_instruction_count_after_bucket == 0u &&
                  vds_next_risk_steps.back().next_bucket_name.empty() &&
                  vds_next_risk_steps.back()
                      .next_bucket_blocking_dimension.empty() &&
                  vds_next_risk_steps.back().next_instruction_name.empty(),
              "expected last VDS next-risk step") ||
      !Expect(append_step->bucket_name == "append_consume" &&
                  append_step->blocking_dimension ==
                      "allocator_or_gds_semantics" &&
                  append_step->risk_rank == 0u &&
                  append_step->instruction_count == 2u &&
                  append_step->remaining_instruction_count_including_bucket ==
                      24u &&
                  append_step->remaining_instruction_count_after_bucket == 22u &&
                  append_step->next_bucket_name ==
                      "exchange_compare_store" &&
                  append_step->next_instruction_name ==
                      "DS_CONDXCHG32_RTN_B64",
              "expected append VDS next-risk step") ||
      !Expect(exchange_step->bucket_name == "exchange_compare_store" &&
                  exchange_step->risk_rank == 1u &&
                  exchange_step->instruction_count == 7u &&
                  exchange_step->remaining_instruction_count_including_bucket ==
                      22u &&
                  exchange_step->remaining_instruction_count_after_bucket ==
                      15u &&
                  exchange_step->next_bucket_name == "multi_address" &&
                  exchange_step->next_bucket_blocking_dimension ==
                      "multi_address_semantics" &&
                  exchange_step->next_instruction_name ==
                      "DS_LOAD_2ADDR_B32",
              "expected exchange/compare-store VDS next-risk step") ||
      !Expect(multi_address_step->bucket_name == "multi_address" &&
                  multi_address_step->risk_rank == 2u &&
                  multi_address_step->instruction_count == 12u &&
                  multi_address_step
                          ->remaining_instruction_count_including_bucket ==
                      15u &&
                  multi_address_step->remaining_instruction_count_after_bucket ==
                      3u &&
                  multi_address_step->next_bucket_name == "bvh_stack" &&
                  multi_address_step->next_bucket_blocking_dimension ==
                      "gfx1201_specific_bvh_semantics" &&
                  multi_address_step->next_instruction_name ==
                      "DS_BVH_STACK_PUSH4_POP1_RTN_B32",
              "expected multi-address VDS next-risk step") ||
      !Expect(bvh_step->bucket_name == "bvh_stack" &&
                  bvh_step->risk_rank == 3u &&
                  bvh_step->instruction_count == 3u &&
                  bvh_step->remaining_instruction_count_including_bucket == 3u &&
                  bvh_step->remaining_instruction_count_after_bucket == 0u &&
                  bvh_step->next_bucket_name.empty() &&
                  bvh_step->next_instruction_name.empty(),
              "expected BVH VDS next-risk step")) {
    return 1;
  }

  if (!Expect(vds_boundary_buckets[0].bucket_name == "append_consume",
              "expected append/consume VDS bucket") ||
      !Expect(vds_boundary_buckets[0].example_instruction == "DS_APPEND",
              "expected append/consume example") ||
      !Expect(vds_boundary_buckets[0].blocking_dimension ==
                  "allocator_or_gds_semantics",
              "expected append/consume blocking dimension") ||
      !Expect(vds_boundary_buckets[0].risk_rank == 0u,
              "expected append/consume risk rank") ||
      !Expect(vds_boundary_buckets[0].starting_instruction_ordinal == 0u &&
                  vds_boundary_buckets[0].ending_instruction_ordinal == 1u,
              "expected append/consume ordinal range") ||
      !Expect(!vds_boundary_buckets[0].safe_under_current_request,
              "expected append/consume to stay outside the safe boundary") ||
      !Expect(vds_boundary_buckets[0].instruction_count == 2u,
              "expected append/consume count") ||
      !Expect(vds_boundary_buckets[0].instruction_names.size() == 2u,
              "expected append/consume instruction list") ||
      !Expect(vds_boundary_buckets[0].instruction_names[0] == "DS_APPEND" &&
                  vds_boundary_buckets[0].instruction_names[1] == "DS_CONSUME",
              "expected append/consume instruction names") ||
      !Expect(vds_boundary_buckets[1].bucket_name == "exchange_compare_store",
              "expected exchange/compare-store VDS bucket") ||
      !Expect(vds_boundary_buckets[1].example_instruction ==
                  "DS_CONDXCHG32_RTN_B64",
              "expected exchange/compare-store example") ||
      !Expect(vds_boundary_buckets[1].blocking_dimension ==
                  "exchange_compare_store_semantics",
              "expected exchange/compare-store blocking dimension") ||
      !Expect(vds_boundary_buckets[1].risk_rank == 1u,
              "expected exchange/compare-store risk rank") ||
      !Expect(vds_boundary_buckets[1].starting_instruction_ordinal == 2u &&
                  vds_boundary_buckets[1].ending_instruction_ordinal == 8u,
              "expected exchange/compare-store ordinal range") ||
      !Expect(!vds_boundary_buckets[1].safe_under_current_request,
              "expected exchange/compare-store to stay outside the safe boundary") ||
      !Expect(vds_boundary_buckets[1].instruction_count == 7u,
              "expected exchange/compare-store count") ||
      !Expect(vds_boundary_buckets[1].instruction_names.size() == 7u,
              "expected exchange/compare-store instruction list") ||
      !Expect(vds_boundary_buckets[1].instruction_names.front() ==
                      "DS_CONDXCHG32_RTN_B64" &&
                  vds_boundary_buckets[1].instruction_names.back() ==
                      "DS_STOREXCHG_RTN_B64",
              "expected exchange/compare-store instruction range") ||
      !Expect(vds_boundary_buckets[2].bucket_name == "multi_address",
              "expected multi-address VDS bucket") ||
      !Expect(vds_boundary_buckets[2].example_instruction ==
                  "DS_LOAD_2ADDR_B32",
              "expected multi-address example") ||
      !Expect(vds_boundary_buckets[2].blocking_dimension ==
                  "multi_address_semantics",
              "expected multi-address blocking dimension") ||
      !Expect(vds_boundary_buckets[2].risk_rank == 2u,
              "expected multi-address risk rank") ||
      !Expect(vds_boundary_buckets[2].starting_instruction_ordinal == 9u &&
                  vds_boundary_buckets[2].ending_instruction_ordinal == 20u,
              "expected multi-address ordinal range") ||
      !Expect(!vds_boundary_buckets[2].safe_under_current_request,
              "expected multi-address to stay outside the safe boundary") ||
      !Expect(vds_boundary_buckets[2].instruction_count == 12u,
              "expected multi-address count") ||
      !Expect(vds_boundary_buckets[2].instruction_names.size() == 12u,
              "expected multi-address instruction list") ||
      !Expect(vds_boundary_buckets[2].instruction_names.front() ==
                      "DS_LOAD_2ADDR_B32" &&
                  vds_boundary_buckets[2].instruction_names.back() ==
                      "DS_STORE_2ADDR_STRIDE64_B64",
              "expected multi-address instruction range") ||
      !Expect(vds_boundary_buckets[3].bucket_name == "bvh_stack",
              "expected BVH-stack VDS bucket") ||
      !Expect(vds_boundary_buckets[3].example_instruction ==
                  "DS_BVH_STACK_PUSH4_POP1_RTN_B32",
              "expected BVH-stack example") ||
      !Expect(vds_boundary_buckets[3].blocking_dimension ==
                  "gfx1201_specific_bvh_semantics",
              "expected BVH-stack blocking dimension") ||
      !Expect(vds_boundary_buckets[3].risk_rank == 3u,
              "expected BVH-stack risk rank") ||
      !Expect(vds_boundary_buckets[3].starting_instruction_ordinal == 21u &&
                  vds_boundary_buckets[3].ending_instruction_ordinal == 23u,
              "expected BVH-stack ordinal range") ||
      !Expect(!vds_boundary_buckets[3].safe_under_current_request,
              "expected BVH-stack to stay outside the safe boundary") ||
      !Expect(vds_boundary_buckets[3].instruction_count == 3u,
              "expected BVH-stack count") ||
      !Expect(vds_boundary_buckets[3].instruction_names.size() == 3u,
              "expected BVH-stack instruction list") ||
      !Expect(vds_boundary_buckets[3].instruction_names[0] ==
                      "DS_BVH_STACK_PUSH4_POP1_RTN_B32" &&
                  vds_boundary_buckets[3].instruction_names[1] ==
                      "DS_BVH_STACK_PUSH8_POP1_RTN_B32" &&
                  vds_boundary_buckets[3].instruction_names[2] ==
                      "DS_BVH_STACK_PUSH8_POP2_RTN_B64",
              "expected BVH-stack instruction names")) {
    return 1;
  }

  if (!Expect(FindGfx1201Wave32Phase0VdsBoundaryBucketForInstruction(
                      "DS_APPEND") == append_consume_bucket &&
                  FindGfx1201Wave32Phase0VdsBoundaryBucketForInstruction(
                      "DS_CONSUME") == append_consume_bucket,
              "expected append/consume instruction lookup") ||
      !Expect(FindGfx1201Wave32Phase0VdsBoundaryBucketForInstruction(
                      "DS_CONDXCHG32_RTN_B64") ==
                      exchange_compare_store_bucket &&
                  FindGfx1201Wave32Phase0VdsBoundaryBucketForInstruction(
                      "DS_STOREXCHG_RTN_B64") ==
                      exchange_compare_store_bucket,
              "expected exchange/compare-store instruction lookup") ||
      !Expect(FindGfx1201Wave32Phase0VdsBoundaryBucketForInstruction(
                      "DS_LOAD_2ADDR_B32") == multi_address_bucket &&
                  FindGfx1201Wave32Phase0VdsBoundaryBucketForInstruction(
                      "DS_STORE_2ADDR_STRIDE64_B64") ==
                      multi_address_bucket,
              "expected multi-address instruction lookup") ||
      !Expect(FindGfx1201Wave32Phase0VdsBoundaryBucketForInstruction(
                      "DS_BVH_STACK_PUSH4_POP1_RTN_B32") ==
                      bvh_stack_bucket &&
                  FindGfx1201Wave32Phase0VdsBoundaryBucketForInstruction(
                      "DS_BVH_STACK_PUSH8_POP2_RTN_B64") == bvh_stack_bucket,
              "expected BVH-stack instruction lookup") ||
      !Expect(FindGfx1201Wave32Phase0VdsBoundaryBucketForInstruction(
                      "DS_BPERMUTE_FI_B32") == nullptr,
              "expected executable VDS op to stay out of boundary buckets") ||
      !Expect(FindGfx1201Wave32Phase0VdsBoundaryBucketForInstruction(
                      "GLOBAL_LOAD_B32") == nullptr,
              "expected non-VDS op to stay out of boundary buckets")) {
    return 1;
  }

  if (!Expect(smem->example_instruction == "S_LOAD_B32",
              "expected ENC_SMEM example instruction") ||
      !Expect(smem->first_executable_instruction == "S_ATC_PROBE",
              "expected ENC_SMEM executable example") ||
      !Expect(vop3->example_instruction == "V_ADD3_U32",
              "expected ENC_VOP3 example instruction") ||
      !Expect(vds->example_instruction == "DS_ADD_U32",
              "expected ENC_VDS example instruction") ||
      !Expect(vds->first_executable_instruction == "DS_ADD_F32",
              "expected ENC_VDS executable example") ||
      !Expect(vglobal->example_instruction == "GLOBAL_LOAD_B32",
              "expected ENC_VGLOBAL example instruction") ||
      !Expect(vglobal->first_executable_instruction == "GLOBAL_ATOMIC_ADD_F32",
              "expected ENC_VGLOBAL executable example")) {
    return 1;
  }

  return 0;
}
