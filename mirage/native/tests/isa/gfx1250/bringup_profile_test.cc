#include <iostream>
#include <string_view>

#include "lib/sim/isa/gfx1250/bringup_profile.h"

namespace {

using mirage::sim::isa::gfx1250::BringupFocusArea;
using mirage::sim::isa::gfx1250::BringupSummary;
using mirage::sim::isa::gfx1250::GetBringupSummary;
using mirage::sim::isa::gfx1250::GetFocusInstructions;
using mirage::sim::isa::gfx1250::IsFocusInstruction;

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

bool Contains(BringupFocusArea area, std::string_view instruction_name) {
  for (const std::string_view candidate : GetFocusInstructions(area)) {
    if (candidate == instruction_name) {
      return true;
    }
  }
  return false;
}

}  // namespace

int main() {
  const BringupSummary& summary = GetBringupSummary();

  if (!Expect(summary.rdna4_instruction_count > 0,
              "expected non-empty RDNA4 instruction count")) {
    return 1;
  }
  if (!Expect(summary.shared_instruction_count > 0,
              "expected shared gfx1250/gfx950 instruction count")) {
    return 1;
  }
  if (!Expect(summary.vop3p_instruction_count > 0,
              "expected non-empty VOP3P bring-up set")) {
    return 1;
  }
  if (!Expect(summary.wmma_instruction_count > 0,
              "expected non-empty WMMA bring-up set")) {
    return 1;
  }
  if (!Expect(summary.fp8_bf8_instruction_count > 0,
              "expected non-empty FP8/BF8 bring-up set")) {
    return 1;
  }
  if (!Expect(summary.scale_paired_instruction_count > 0,
              "expected non-empty scale/paired bring-up set")) {
    return 1;
  }

  if (!Expect(Contains(BringupFocusArea::kVop3p, "V_PK_ADD_BF16"),
              "expected V_PK_ADD_BF16 in VOP3P focus set")) {
    return 1;
  }
  if (!Expect(
          Contains(BringupFocusArea::kWmma, "V_WMMA_F32_16X16X4_F32_w32"),
          "expected V_WMMA_F32_16X16X4_F32_w32 in WMMA focus set")) {
    return 1;
  }
  if (!Expect(Contains(BringupFocusArea::kFp8Bf8, "V_CVT_F16_FP8"),
              "expected V_CVT_F16_FP8 in FP8/BF8 focus set")) {
    return 1;
  }
  if (!Expect(
          Contains(BringupFocusArea::kScalePaired, "V_WMMA_LD_SCALE_PAIRED_B32"),
          "expected V_WMMA_LD_SCALE_PAIRED_B32 in scale/paired focus set")) {
    return 1;
  }
  if (!Expect(IsFocusInstruction("V_CVT_F16_FP8"),
              "expected IsFocusInstruction to match FP8/BF8 op")) {
    return 1;
  }
  return 0;
}
