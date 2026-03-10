#ifndef MIRAGE_SIM_ISA_GFX1250_DECODER_SEED_CATALOG_H_
#define MIRAGE_SIM_ISA_GFX1250_DECODER_SEED_CATALOG_H_

#include <cstdint>
#include <span>
#include <string_view>

namespace mirage::sim::isa::gfx1250 {

enum class SeedFamily {
  kVop3p,
  kWmma,
  kFp8Bf8,
  kScalePaired,
};

enum class SeedProvenance {
  kLlvmOnly,
  kRdna4XmlOnly,
  kRdna4AndLlvm,
};

enum class DecodeSeedHint {
  kUnknown,
  kVop1,
  kVop3,
  kVop3p,
  kVop3Sdst,
  kMimgTensor,
};

struct DecoderSeedInfo {
  std::string_view instruction_name{};
  std::string_view llvm_file{};
  std::uint32_t llvm_line = 0;
  std::string_view rdna4_encoding_name{};
  std::uint32_t rdna4_opcode = 0;
  std::uint32_t rdna4_operand_count = 0;
  SeedProvenance provenance = SeedProvenance::kLlvmOnly;
  DecodeSeedHint decode_hint = DecodeSeedHint::kUnknown;
  bool in_vop3p = false;
  bool in_wmma = false;
  bool in_fp8_bf8 = false;
  bool in_scale_paired = false;
  bool appears_in_rdna4_xml = false;
  bool is_target_specific = false;
};

struct SeedFamilyManifest {
  SeedFamily family = SeedFamily::kVop3p;
  std::string_view family_name{};
  std::uint32_t seeded_instruction_count = 0;
  std::uint32_t xml_backed_count = 0;
  std::uint32_t llvm_only_count = 0;
  std::uint32_t target_specific_count = 0;
  std::uint32_t vop1_hint_count = 0;
  std::uint32_t vop3_hint_count = 0;
  std::uint32_t vop3p_hint_count = 0;
  std::uint32_t vop3_sdst_hint_count = 0;
  std::uint32_t mimg_tensor_hint_count = 0;
};

std::span<const DecoderSeedInfo> GetDecoderSeedInfos();
std::span<const std::string_view> GetSeededInstructionNames(SeedFamily family);
const DecoderSeedInfo* FindDecoderSeedInfo(std::string_view instruction_name);
bool HasDecodeSeedFamily(std::string_view instruction_name, SeedFamily family);
std::span<const SeedFamilyManifest> GetSeedFamilyManifests();
const SeedFamilyManifest* FindSeedFamilyManifest(SeedFamily family);

}  // namespace mirage::sim::isa::gfx1250

#endif  // MIRAGE_SIM_ISA_GFX1250_DECODER_SEED_CATALOG_H_
