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

  if (!Expect(statuses.size() == 3u,
              "expected three tracked wave32-local encodings") ||
      !Expect(vop1 != nullptr, "expected ENC_VOP1 status") ||
      !Expect(vop2 != nullptr, "expected ENC_VOP2 status") ||
      !Expect(vopc != nullptr, "expected ENC_VOPC status") ||
      !Expect(missing == nullptr, "expected no status for ENC_VIMAGE")) {
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

  const auto next_risk_encodings = GetGfx1201Wave32Phase0NextRiskEncodings();
  if (!Expect(next_risk_encodings.size() == kExpectedNextRiskEncodings.size(),
              "expected next-risk encoding count")) {
    return 1;
  }

  for (std::size_t i = 0; i < kExpectedNextRiskEncodings.size(); ++i) {
    if (!Expect(next_risk_encodings[i] == kExpectedNextRiskEncodings[i],
                "unexpected next-risk encoding order")) {
      return 1;
    }
  }

  return 0;
}
