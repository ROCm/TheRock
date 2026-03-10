#ifndef MIRAGE_SIM_ISA_GFX1250_BRINGUP_PROFILE_H_
#define MIRAGE_SIM_ISA_GFX1250_BRINGUP_PROFILE_H_

#include <cstddef>
#include <span>
#include <string_view>

namespace mirage::sim::isa::gfx1250 {

enum class BringupFocusArea {
  kVop3p,
  kWmma,
  kFp8Bf8,
  kScalePaired,
};

struct BringupSummary {
  std::size_t rdna4_instruction_count = 0;
  std::size_t gfx950_instruction_count = 0;
  std::size_t shared_instruction_count = 0;
  std::size_t rdna4_only_instruction_count = 0;
  std::size_t gfx950_only_instruction_count = 0;
  std::size_t llvm_normalized_symbol_count = 0;
  std::size_t llvm_target_specific_count = 0;
  std::size_t vop3p_instruction_count = 0;
  std::size_t wmma_instruction_count = 0;
  std::size_t fp8_bf8_instruction_count = 0;
  std::size_t scale_paired_instruction_count = 0;
};

const BringupSummary& GetBringupSummary();
std::span<const std::string_view> GetFocusInstructions(BringupFocusArea area);
bool IsFocusInstruction(std::string_view instruction_name);

}  // namespace mirage::sim::isa::gfx1250

#endif  // MIRAGE_SIM_ISA_GFX1250_BRINGUP_PROFILE_H_
