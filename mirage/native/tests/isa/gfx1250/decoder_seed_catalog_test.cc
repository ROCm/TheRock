#include <iostream>
#include <string_view>

#include "lib/sim/isa/gfx1250/decoder_seed_catalog.h"

namespace {

using mirage::sim::isa::gfx1250::DecodeSeedHint;
using mirage::sim::isa::gfx1250::DecoderSeedInfo;
using mirage::sim::isa::gfx1250::FindDecoderSeedInfo;
using mirage::sim::isa::gfx1250::FindSeedFamilyManifest;
using mirage::sim::isa::gfx1250::GetDecoderSeedInfos;
using mirage::sim::isa::gfx1250::GetSeedFamilyManifests;
using mirage::sim::isa::gfx1250::GetSeededInstructionNames;
using mirage::sim::isa::gfx1250::HasDecodeSeedFamily;
using mirage::sim::isa::gfx1250::SeedFamily;
using mirage::sim::isa::gfx1250::SeedFamilyManifest;
using mirage::sim::isa::gfx1250::SeedProvenance;

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

bool Contains(SeedFamily family, std::string_view instruction_name) {
  for (const std::string_view candidate : GetSeededInstructionNames(family)) {
    if (candidate == instruction_name) {
      return true;
    }
  }
  return false;
}

}  // namespace

int main() {
  if (!Expect(!GetDecoderSeedInfos().empty(),
              "expected non-empty gfx1250 decoder seed table")) {
    return 1;
  }
  if (!Expect(GetSeedFamilyManifests().size() == 4,
              "expected four gfx1250 seed family manifests")) {
    return 1;
  }

  const DecoderSeedInfo* vop3p = FindDecoderSeedInfo("V_PK_ADD_BF16");
  if (!Expect(vop3p != nullptr, "expected VOP3P seed lookup")) {
    return 1;
  }
  if (!Expect(vop3p->decode_hint == DecodeSeedHint::kVop3p,
              "expected VOP3P decode hint for V_PK_ADD_BF16")) {
    return 1;
  }
  if (!Expect(vop3p->provenance == SeedProvenance::kLlvmOnly,
              "expected LLVM-only provenance for V_PK_ADD_BF16")) {
    return 1;
  }
  if (!Expect(HasDecodeSeedFamily("V_PK_ADD_BF16", SeedFamily::kVop3p),
              "expected V_PK_ADD_BF16 in VOP3P seed family")) {
    return 1;
  }

  const DecoderSeedInfo* tensor = FindDecoderSeedInfo("TENSOR_LOAD_TO_LDS");
  if (!Expect(tensor != nullptr, "expected tensor seed lookup")) {
    return 1;
  }
  if (!Expect(tensor->decode_hint == DecodeSeedHint::kMimgTensor,
              "expected tensor decode hint for TENSOR_LOAD_TO_LDS")) {
    return 1;
  }
  if (!Expect(HasDecodeSeedFamily("TENSOR_LOAD_TO_LDS", SeedFamily::kWmma),
              "expected tensor op in WMMA seed family")) {
    return 1;
  }

  const DecoderSeedInfo* fp8 = FindDecoderSeedInfo("V_CVT_F32_FP8");
  if (!Expect(fp8 != nullptr, "expected FP8 seed lookup")) {
    return 1;
  }
  if (!Expect(fp8->appears_in_rdna4_xml,
              "expected V_CVT_F32_FP8 to be XML-backed")) {
    return 1;
  }
  if (!Expect(fp8->provenance == SeedProvenance::kRdna4AndLlvm,
              "expected mixed provenance for V_CVT_F32_FP8")) {
    return 1;
  }
  if (!Expect(fp8->decode_hint == DecodeSeedHint::kVop1,
              "expected VOP1 decode hint for V_CVT_F32_FP8")) {
    return 1;
  }

  const DecoderSeedInfo* scale = FindDecoderSeedInfo("V_DIV_SCALE_F64");
  if (!Expect(scale != nullptr, "expected XML-backed scale seed lookup")) {
    return 1;
  }
  if (!Expect(scale->decode_hint == DecodeSeedHint::kVop3Sdst,
              "expected VOP3 SDST decode hint for scale conversion")) {
    return 1;
  }
  if (!Expect(HasDecodeSeedFamily("V_DIV_SCALE_F64", SeedFamily::kScalePaired),
              "expected XML-backed scale op in scale/paired seed family")) {
    return 1;
  }

  const DecoderSeedInfo* paired =
      FindDecoderSeedInfo("V_WMMA_LD_SCALE_PAIRED_B32");
  if (!Expect(paired != nullptr, "expected paired-load scale seed lookup")) {
    return 1;
  }
  if (!Expect(paired->decode_hint == DecodeSeedHint::kVop3p,
              "expected VOP3P hint for paired scale load")) {
    return 1;
  }

  if (!Expect(Contains(SeedFamily::kWmma, "V_WMMA_F32_16X16X4_F32_w32"),
              "expected WMMA instruction in WMMA seed list")) {
    return 1;
  }
  if (!Expect(Contains(SeedFamily::kFp8Bf8, "V_CVT_F16_FP8"),
              "expected FP8 conversion in FP8/BF8 seed list")) {
    return 1;
  }

  const SeedFamilyManifest* wmma_manifest =
      FindSeedFamilyManifest(SeedFamily::kWmma);
  if (!Expect(wmma_manifest != nullptr, "expected WMMA family manifest")) {
    return 1;
  }
  if (!Expect(wmma_manifest->mimg_tensor_hint_count > 0,
              "expected tensor-backed WMMA seed coverage")) {
    return 1;
  }
  if (!Expect(wmma_manifest->vop3p_hint_count > 0,
              "expected VOP3P-backed WMMA seed coverage")) {
    return 1;
  }

  const SeedFamilyManifest* scale_manifest =
      FindSeedFamilyManifest(SeedFamily::kScalePaired);
  if (!Expect(scale_manifest != nullptr,
              "expected scale/paired family manifest")) {
    return 1;
  }
  if (!Expect(scale_manifest->xml_backed_count > 0,
              "expected XML-backed scale/paired seeds")) {
    return 1;
  }
  if (!Expect(scale_manifest->target_specific_count > 0,
              "expected target-specific scale/paired seeds")) {
    return 1;
  }

  return 0;
}
