#include <iostream>

#include "lib/sim/isa/gfx1201/binary_decoder.h"
#include "lib/sim/isa/gfx1201/decoder_seed.h"

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

  const auto seeds = GetGfx1201Phase0ComputeDecoderSeeds();
  if (!Expect(seeds.size() == 12u, "expected 12 phase-0 compute encodings") ||
      !Expect(GetGfx1201Phase0ComputeDecoderSeedEntries().size() == 1198u,
              "expected phase-0 compute seed entry count")) {
    return 1;
  }

  const Gfx1201DecoderSeedEncoding* enc_sopp =
      FindGfx1201Phase0ComputeDecoderSeed("ENC_SOPP");
  const Gfx1201DecoderSeedEncoding* enc_sop1 =
      FindGfx1201Phase0ComputeDecoderSeed("ENC_SOP1");
  const Gfx1201DecoderSeedEncoding* enc_vopc =
      FindGfx1201Phase0ComputeDecoderSeed("ENC_VOPC");
  const Gfx1201DecoderSeedEncoding* enc_vop3 =
      FindGfx1201Phase0ComputeDecoderSeed("ENC_VOP3");
  const Gfx1201DecoderSeedEncoding* enc_vglobal =
      FindGfx1201Phase0ComputeDecoderSeed("ENC_VGLOBAL");
  const Gfx1201DecoderSeedEncoding* missing =
      FindGfx1201Phase0ComputeDecoderSeed("ENC_VIMAGE");

  if (!Expect(enc_sopp != nullptr, "expected ENC_SOPP seed") ||
      !Expect(enc_sop1 != nullptr, "expected ENC_SOP1 seed") ||
      !Expect(enc_vopc != nullptr, "expected ENC_VOPC seed") ||
      !Expect(enc_vop3 != nullptr, "expected ENC_VOP3 seed") ||
      !Expect(enc_vglobal != nullptr, "expected ENC_VGLOBAL seed") ||
      !Expect(missing == nullptr, "expected graphics encoding to be absent")) {
    return 1;
  }

  if (!Expect(enc_sopp->instruction_count == 41u,
              "expected ENC_SOPP instruction count") ||
      !Expect(enc_sopp->transferable_as_is_count == 8u,
              "expected ENC_SOPP as-is count") ||
      !Expect(enc_sopp->TransferableWithDecoderWorkRollupCount() == 14u,
              "expected ENC_SOPP decoder rollup count") ||
      !Expect(enc_sopp->gfx1201_specific_count == 19u,
              "expected ENC_SOPP gfx1201-specific count") ||
      !Expect(enc_sop1->instruction_count == 84u,
              "expected ENC_SOP1 instruction count") ||
      !Expect(enc_sop1->transferable_with_semantic_work_count == 10u,
              "expected ENC_SOP1 semantic-work count") ||
      !Expect(enc_sop1->gfx1201_specific_count == 46u,
              "expected ENC_SOP1 gfx1201-specific count") ||
      !Expect(enc_vopc->transferable_with_decoder_work_count == 30u,
              "expected ENC_VOPC decoder-only count") ||
      !Expect(enc_vopc->transferable_with_decoder_and_semantic_work_count == 24u,
              "expected ENC_VOPC dual-work count") ||
      !Expect(enc_vop3->instruction_count == 434u,
              "expected ENC_VOP3 instruction count") ||
      !Expect(enc_vop3->alternate_entry_count == 0u,
              "expected ENC_VOP3 seed to only track ENC_VOP3 entries") ||
      !Expect(enc_vglobal->instruction_count == 65u,
              "expected ENC_VGLOBAL instruction count") ||
      !Expect(enc_vglobal->transferable_as_is_count == 3u,
              "expected ENC_VGLOBAL as-is count") ||
      !Expect(enc_vglobal->gfx1201_specific_count == 62u,
              "expected ENC_VGLOBAL gfx1201-specific count")) {
    return 1;
  }

  const auto sopp_entries = GetGfx1201Phase0ComputeDecoderSeedEntries(*enc_sopp);
  const auto vop3_entries = GetGfx1201Phase0ComputeDecoderSeedEntries(*enc_vop3);
  const Gfx1201DecoderSeedEntry* s_endpgm_entry = nullptr;
  for (const Gfx1201DecoderSeedEntry& entry : sopp_entries) {
    if (entry.instruction_name == "S_ENDPGM") {
      s_endpgm_entry = &entry;
      break;
    }
  }
  if (!Expect(sopp_entries.size() == 41u, "expected ENC_SOPP entry count") ||
      !Expect(vop3_entries.size() == 434u, "expected ENC_VOP3 entry count") ||
      !Expect(s_endpgm_entry != nullptr, "expected S_ENDPGM seed entry") ||
      !Expect(s_endpgm_entry->opcode == 48u,
              "expected S_ENDPGM opcode") ||
      !Expect(s_endpgm_entry->is_default_encoding,
              "expected S_ENDPGM default encoding") ||
      !Expect(vop3_entries.front().encoding_name == "ENC_VOP3",
              "expected ENC_VOP3 seed entries")) {
    return 1;
  }

  if (!Expect(IsGfx1201Phase0ComputeEncoding("ENC_VDS"),
              "expected ENC_VDS to be phase-0 compute") ||
      !Expect(!IsGfx1201Phase0ComputeEncoding("ENC_VIMAGE"),
              "expected ENC_VIMAGE to be out of scope")) {
    return 1;
  }

  Gfx1201BinaryDecoder decoder;
  if (!Expect(decoder.Phase0ComputeSeeds().size() == seeds.size(),
              "expected decoder to expose phase-0 seeds") ||
      !Expect(decoder.FindPhase0ComputeSeed("ENC_VOP3") == enc_vop3,
              "expected decoder seed lookup to delegate")) {
    return 1;
  }

  return 0;
}
