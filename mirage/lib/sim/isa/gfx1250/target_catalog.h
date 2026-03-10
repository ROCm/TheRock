#ifndef MIRAGE_SIM_ISA_GFX1250_TARGET_CATALOG_H_
#define MIRAGE_SIM_ISA_GFX1250_TARGET_CATALOG_H_

#include <cstdint>
#include <span>
#include <string_view>

namespace mirage::sim::isa::gfx1250 {

struct TargetOpcodeInfo {
  std::string_view instruction_name{};
  std::string_view llvm_file{};
  std::uint32_t llvm_line = 0;
  bool appears_in_rdna4_xml = false;
  bool is_target_specific = false;
  bool is_vop3 = false;
  bool is_vop3p = false;
  bool is_wmma = false;
  bool is_fp8_bf8 = false;
  bool is_scale_paired = false;
};

std::span<const TargetOpcodeInfo> GetTargetOpcodeInfos();
const TargetOpcodeInfo* FindTargetOpcodeInfo(std::string_view instruction_name);
std::span<const std::string_view> GetSharedInstructionSample();
std::span<const std::string_view> GetRdna4OnlyInstructionSample();

}  // namespace mirage::sim::isa::gfx1250

#endif  // MIRAGE_SIM_ISA_GFX1250_TARGET_CATALOG_H_
