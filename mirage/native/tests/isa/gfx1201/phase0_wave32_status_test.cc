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

  if (!Expect(statuses.size() == 3u,
              "expected three tracked wave32-local encodings") ||
      !Expect(vop1 != nullptr, "expected ENC_VOP1 status") ||
      !Expect(vop2 != nullptr, "expected ENC_VOP2 status") ||
      !Expect(vopc != nullptr, "expected ENC_VOPC status") ||
      !Expect(missing == nullptr, "expected no status for ENC_VIMAGE") ||
      !Expect(next_risk_statuses.size() == 4u,
              "expected four next-risk encoding statuses") ||
      !Expect(smem != nullptr, "expected ENC_SMEM next-risk status") ||
      !Expect(vop3 != nullptr, "expected ENC_VOP3 next-risk status") ||
      !Expect(vds != nullptr, "expected ENC_VDS next-risk status") ||
      !Expect(vglobal != nullptr, "expected ENC_VGLOBAL next-risk status")) {
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
      !Expect(vopc->fully_executable, "expected ENC_VOPC saturation")) {
    return 1;
  }

  if (!Expect(smem->seeded_instruction_count == 28u,
              "expected ENC_SMEM seeded instruction count") ||
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
      !Expect(vds->TransferableWithDecoderRollupCount() == 38u,
              "expected ENC_VDS decoder rollup count") ||
      !Expect(vds->gfx1201_specific_count == 58u,
              "expected ENC_VDS gfx1201-specific count") ||
      !Expect(vglobal->seeded_instruction_count == 65u,
              "expected ENC_VGLOBAL seeded instruction count") ||
      !Expect(vglobal->transferable_as_is_count == 3u,
              "expected ENC_VGLOBAL as-is count") ||
      !Expect(vglobal->gfx1201_specific_count == 62u,
              "expected ENC_VGLOBAL gfx1201-specific count")) {
    return 1;
  }

  if (!Expect(decoder.Phase0ExecutableOpcodes().size() == 325u,
              "expected phase-0 executable opcode count") ||
      !Expect(IsGfx1201Wave32Phase0EncodingSaturated("ENC_VOP1"),
              "expected ENC_VOP1 saturation helper") ||
      !Expect(IsGfx1201Wave32Phase0EncodingSaturated("ENC_VOP2"),
              "expected ENC_VOP2 saturation helper") ||
      !Expect(IsGfx1201Wave32Phase0EncodingSaturated("ENC_VOPC"),
              "expected ENC_VOPC saturation helper") ||
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

  const auto next_risk_encodings = GetGfx1201Wave32Phase0NextRiskEncodings();
  const auto frontier_order = GetGfx1201Wave32Phase0FrontierOrder();
  if (!Expect(next_risk_encodings.size() == kExpectedNextRiskEncodings.size(),
              "expected next-risk encoding count") ||
      !Expect(frontier_order.size() == kExpectedFrontierOrder.size(),
              "expected frontier order count") ||
      !Expect(GetGfx1201Wave32Phase0RecommendedNextEncoding() == "ENC_SMEM",
              "expected ENC_SMEM as the recommended next frontier")) {
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

  if (!Expect(smem->example_instruction == "S_LOAD_B32",
              "expected ENC_SMEM example instruction") ||
      !Expect(vop3->example_instruction == "V_ADD3_U32",
              "expected ENC_VOP3 example instruction") ||
      !Expect(vds->example_instruction == "DS_ADD_U32",
              "expected ENC_VDS example instruction") ||
      !Expect(vglobal->example_instruction == "GLOBAL_LOAD_B32",
              "expected ENC_VGLOBAL example instruction")) {
    return 1;
  }

  return 0;
}
