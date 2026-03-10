#include <iostream>

#include "lib/sim/isa/gfx1201/support_catalog.h"

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

  const InstructionCatalogMetadata& metadata = GetGfx1201SupportCatalogMetadata();
  if (!Expect(metadata.gfx_target == "gfx1201", "expected gfx1201 metadata") ||
      !Expect(metadata.instruction_count == 1264u,
              "expected imported instruction count") ||
      !Expect(metadata.encoding_count == 5062u,
              "expected imported encoding count")) {
    return 1;
  }

  const auto rollup_summaries = GetGfx1201SupportRollupSummaries();
  if (!Expect(rollup_summaries.size() == 4u,
              "expected four rollup summaries") ||
      !Expect(rollup_summaries[0].instruction_count == 363u,
              "expected transferable-as-is count") ||
      !Expect(rollup_summaries[1].instruction_count == 197u,
              "expected decoder-work count") ||
      !Expect(rollup_summaries[2].instruction_count == 36u,
              "expected semantic-work count") ||
      !Expect(rollup_summaries[3].instruction_count == 668u,
              "expected gfx1201-specific count")) {
    return 1;
  }

  const auto state_summaries = GetGfx1201SupportStateSummaries();
  if (!Expect(state_summaries.size() == 5u, "expected five state summaries") ||
      !Expect(state_summaries[3].instruction_count == 167u,
              "expected decoder+semantic-work count")) {
    return 1;
  }

  const Gfx1201InstructionSupportInfo* s_mov =
      FindGfx1201InstructionSupport("S_MOV_B32");
  const Gfx1201InstructionSupportInfo* v_cmp_lt_f16 =
      FindGfx1201InstructionSupport("V_CMP_LT_F16");
  const Gfx1201InstructionSupportInfo* v_cvt_f32_fp8 =
      FindGfx1201InstructionSupport("V_CVT_F32_FP8");
  const Gfx1201InstructionSupportInfo* s_nop =
      FindGfx1201InstructionSupport("S_NOP");
  const Gfx1201InstructionSupportInfo* image_load =
      FindGfx1201InstructionSupport("IMAGE_LOAD");
  const Gfx1201InstructionSupportInfo* missing =
      FindGfx1201InstructionSupport("MIRAGE_FAKE_OPCODE");

  if (!Expect(s_mov != nullptr, "expected S_MOV_B32") ||
      !Expect(v_cmp_lt_f16 != nullptr, "expected V_CMP_LT_F16") ||
      !Expect(v_cvt_f32_fp8 != nullptr, "expected V_CVT_F32_FP8") ||
      !Expect(s_nop != nullptr, "expected S_NOP") ||
      !Expect(image_load != nullptr, "expected IMAGE_LOAD") ||
      !Expect(missing == nullptr, "expected missing opcode lookup to fail")) {
    return 1;
  }

  if (!Expect(s_mov->rollup == Gfx1201SupportRollup::kTransferableAsIs,
              "expected S_MOV_B32 to transfer as-is") ||
      !Expect(!NeedsDecoderWork(*s_mov) && !NeedsSemanticWork(*s_mov),
              "expected S_MOV_B32 to need no extra work") ||
      !Expect(v_cmp_lt_f16->state ==
                  Gfx1201SupportState::kTransferableWithDecoderWork,
              "expected V_CMP_LT_F16 to need decoder work") ||
      !Expect(NeedsDecoderWork(*v_cmp_lt_f16) &&
                  !NeedsSemanticWork(*v_cmp_lt_f16),
              "expected V_CMP_LT_F16 decoder-only gap") ||
      !Expect(v_cvt_f32_fp8->state ==
                  Gfx1201SupportState::kTransferableWithSemanticWork,
              "expected V_CVT_F32_FP8 to need semantic work") ||
      !Expect(!NeedsDecoderWork(*v_cvt_f32_fp8) &&
                  NeedsSemanticWork(*v_cvt_f32_fp8),
              "expected V_CVT_F32_FP8 semantic-only gap") ||
      !Expect(s_nop->state ==
                  Gfx1201SupportState::kTransferableWithDecoderAndSemanticWork,
              "expected S_NOP to need decoder and semantic work") ||
      !Expect(NeedsDecoderWork(*s_nop) && NeedsSemanticWork(*s_nop),
              "expected S_NOP dual work gap") ||
      !Expect(image_load->rollup == Gfx1201SupportRollup::kGfx1201Specific,
              "expected IMAGE_LOAD to be gfx1201-specific") ||
      !Expect(!image_load->known_in_gfx950_catalog,
              "expected IMAGE_LOAD to be absent from gfx950")) {
    return 1;
  }

  if (!Expect(GetGfx1201InstructionsByRollup(
                  Gfx1201SupportRollup::kTransferableWithDecoderWork)
                  .size() == 197u,
              "expected decoder-work rollup size") ||
      !Expect(GetGfx1201InstructionsByState(
                  Gfx1201SupportState::kTransferableWithDecoderAndSemanticWork)
                  .size() == 167u,
              "expected dual-work state size") ||
      !Expect(!GetGfx1201Encodings(*image_load).empty(),
              "expected IMAGE_LOAD encodings") ||
      !Expect(GetGfx1201InstructionSupportCatalog().size() == 1264u,
              "expected full support catalog size") ||
      !Expect(GetGfx1201InstructionSupportEncodings().size() == 5062u,
              "expected full encoding catalog size")) {
    return 1;
  }

  return 0;
}
