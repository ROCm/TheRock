#include <iostream>
#include <string_view>

#include "lib/sim/isa/gfx1250/target_catalog.h"

namespace {

using mirage::sim::isa::gfx1250::FindTargetOpcodeInfo;
using mirage::sim::isa::gfx1250::GetRdna4OnlyInstructionSample;
using mirage::sim::isa::gfx1250::GetSharedInstructionSample;
using mirage::sim::isa::gfx1250::GetTargetOpcodeInfos;
using mirage::sim::isa::gfx1250::TargetOpcodeInfo;

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!Expect(!GetTargetOpcodeInfos().empty(),
              "expected non-empty gfx1250 target opcode table")) {
    return 1;
  }
  if (!Expect(!GetSharedInstructionSample().empty(),
              "expected non-empty shared instruction sample")) {
    return 1;
  }
  if (!Expect(!GetRdna4OnlyInstructionSample().empty(),
              "expected non-empty RDNA4-only instruction sample")) {
    return 1;
  }

  const TargetOpcodeInfo* wmma =
      FindTargetOpcodeInfo("V_WMMA_F32_16X16X4_F32_w32");
  if (!Expect(wmma != nullptr, "expected WMMA opcode lookup")) {
    return 1;
  }
  if (!Expect(wmma->is_wmma, "expected WMMA flag on WMMA opcode")) {
    return 1;
  }

  const TargetOpcodeInfo* fp8 = FindTargetOpcodeInfo("V_CVT_F16_FP8");
  if (!Expect(fp8 != nullptr, "expected FP8 opcode lookup")) {
    return 1;
  }
  if (!Expect(fp8->is_fp8_bf8, "expected FP8/BF8 flag on FP8 opcode")) {
    return 1;
  }

  const TargetOpcodeInfo* scale =
      FindTargetOpcodeInfo("V_WMMA_LD_SCALE_PAIRED_B32");
  if (!Expect(scale != nullptr, "expected scale/paired opcode lookup")) {
    return 1;
  }
  if (!Expect(scale->is_scale_paired,
              "expected scale/paired flag on paired load opcode")) {
    return 1;
  }

  const TargetOpcodeInfo* shared =
      FindTargetOpcodeInfo("V_CVT_PK_FP8_F32");
  if (!Expect(shared != nullptr, "expected shared RDNA4 opcode lookup")) {
    return 1;
  }
  if (!Expect(shared->appears_in_rdna4_xml,
              "expected shared RDNA4 opcode to be marked present")) {
    return 1;
  }

  return 0;
}
