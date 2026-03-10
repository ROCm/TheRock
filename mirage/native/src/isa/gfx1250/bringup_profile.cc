#include "lib/sim/isa/gfx1250/bringup_profile.h"

#include <array>

namespace mirage::sim::isa::gfx1250 {
namespace {

constexpr BringupSummary kBringupSummary{
    1264,
    1242,
    596,
    668,
    646,
    172,
    154,
    62,
    47,
    87,
    52,
};

constexpr std::array<std::string_view, 24> kVop3pInstructions{{
    "V_PK_ADD_BF16",
    "V_PK_FMA_BF16",
    "V_PK_ADD_MAX_I16",
    "V_PK_ADD_MAX_U16",
    "V_PK_ADD_MIN_I16",
    "V_PK_ADD_MIN_U16",
    "V_PK_MAX3_I16",
    "V_PK_MAX3_NUM_F16",
    "V_PK_MAX3_U16",
    "V_PK_MAXIMUM3_F16",
    "V_PK_MAX_NUM_BF16",
    "V_PK_MIN3_I16",
    "V_PK_MIN3_NUM_F16",
    "V_PK_MIN3_U16",
    "V_PK_MINIMUM3_F16",
    "V_PK_MIN_NUM_BF16",
    "V_PK_MUL_BF16",
    "V_SWMMAC_BF16F32_16X16X64_BF16_w32",
    "V_SWMMAC_BF16_16X16X64_BF16_w32",
    "V_SWMMAC_F16_16X16X128_BF8_BF8_w32",
    "V_SWMMAC_F16_16X16X128_BF8_FP8_w32",
    "V_SWMMAC_F16_16X16X128_FP8_BF8_w32",
    "V_SWMMAC_F16_16X16X128_FP8_FP8_w32",
    "V_SWMMAC_F16_16X16X64_F16_w32",
}};
constexpr std::array<std::string_view, 24> kWmmaInstructions{{
    "V_WMMA_F32_16X16X4_F32_w32",
    "V_WMMA_BF16F32_16X16X32_BF16_w32",
    "V_SWMMAC_F32_16X16X64_F16_w32",
    "TENSOR_LOAD_TO_LDS",
    "TENSOR_STORE_FROM_LDS",
    "V_SWMMAC_BF16F32_16X16X64_BF16_w32",
    "V_SWMMAC_BF16_16X16X64_BF16_w32",
    "V_SWMMAC_F16_16X16X128_BF8_BF8_w32",
    "V_SWMMAC_F16_16X16X128_BF8_FP8_w32",
    "V_SWMMAC_F16_16X16X128_FP8_BF8_w32",
    "V_SWMMAC_F16_16X16X128_FP8_FP8_w32",
    "V_SWMMAC_F16_16X16X64_F16_w32",
    "V_SWMMAC_F32_16X16X128_BF8_BF8_w32",
    "V_SWMMAC_F32_16X16X128_BF8_FP8_w32",
    "V_SWMMAC_F32_16X16X128_FP8_BF8_w32",
    "V_SWMMAC_F32_16X16X128_FP8_FP8_w32",
    "V_SWMMAC_F32_16X16X64_BF16_w32",
    "V_SWMMAC_I32_16X16X128_IU8_w32",
    "V_WMMA_BF16_16X16X32_BF16_w32",
    "V_WMMA_F16_16X16X128_BF8_BF8_w32",
    "V_WMMA_F16_16X16X128_BF8_FP8_w32",
    "V_WMMA_F16_16X16X128_FP8_BF8_w32",
    "V_WMMA_F16_16X16X128_FP8_FP8_w32",
    "V_WMMA_F16_16X16X32_F16_w32",
}};
constexpr std::array<std::string_view, 32> kFp8Bf8Instructions{{
    "V_CVT_F16_FP8",
    "V_CVT_F16_BF8",
    "V_CVT_PK_FP8_F16",
    "V_CVT_PK_BF8_F16",
    "V_CVT_PK_F16_BF8",
    "V_CVT_PK_F16_FP8",
    "V_CVT_SCALEF32_PK16_BF6_BF16",
    "V_CVT_SCALEF32_PK16_BF6_F16",
    "V_CVT_SCALEF32_PK16_BF6_F32",
    "V_CVT_SCALEF32_PK16_FP6_BF16",
    "V_CVT_SCALEF32_PK16_FP6_F16",
    "V_CVT_SCALEF32_PK16_FP6_F32",
    "V_CVT_SCALEF32_PK8_BF8_BF16",
    "V_CVT_SCALEF32_PK8_BF8_F16",
    "V_CVT_SCALEF32_PK8_BF8_F32",
    "V_CVT_SCALEF32_PK8_FP4_BF16",
    "V_CVT_SCALEF32_PK8_FP4_F16",
    "V_CVT_SCALEF32_PK8_FP4_F32",
    "V_CVT_SCALEF32_PK8_FP8_BF16",
    "V_CVT_SCALEF32_PK8_FP8_F16",
    "V_CVT_SCALEF32_PK8_FP8_F32",
    "V_CVT_SCALEF32_SR_PK16_BF6_BF16",
    "V_CVT_SCALEF32_SR_PK16_BF6_F16",
    "V_CVT_SCALEF32_SR_PK16_BF6_F32",
    "V_CVT_SCALEF32_SR_PK16_FP6_BF16",
    "V_CVT_SCALEF32_SR_PK16_FP6_F16",
    "V_CVT_SCALEF32_SR_PK16_FP6_F32",
    "V_CVT_SCALEF32_SR_PK8_BF8_BF16",
    "V_CVT_SCALEF32_SR_PK8_BF8_F16",
    "V_CVT_SCALEF32_SR_PK8_BF8_F32",
    "V_CVT_SCALEF32_SR_PK8_FP4_BF16",
    "V_CVT_SCALEF32_SR_PK8_FP4_F16",
}};
constexpr std::array<std::string_view, 24> kScalePairedInstructions{{
    "V_WMMA_LD_SCALE_PAIRED_B32",
    "V_WMMA_LD_SCALE16_PAIRED_B64",
    "V_CVT_SCALEF32_PK16_BF6_BF16",
    "V_CVT_SCALEF32_PK16_BF6_F16",
    "V_CVT_SCALEF32_PK16_BF6_F32",
    "V_CVT_SCALEF32_PK16_FP6_BF16",
    "V_CVT_SCALEF32_PK16_FP6_F16",
    "V_CVT_SCALEF32_PK16_FP6_F32",
    "V_CVT_SCALEF32_PK8_BF8_BF16",
    "V_CVT_SCALEF32_PK8_BF8_F16",
    "V_CVT_SCALEF32_PK8_BF8_F32",
    "V_CVT_SCALEF32_PK8_FP4_BF16",
    "V_CVT_SCALEF32_PK8_FP4_F16",
    "V_CVT_SCALEF32_PK8_FP4_F32",
    "V_CVT_SCALEF32_PK8_FP8_BF16",
    "V_CVT_SCALEF32_PK8_FP8_F16",
    "V_CVT_SCALEF32_PK8_FP8_F32",
    "V_CVT_SCALEF32_SR_PK16_BF6_BF16",
    "V_CVT_SCALEF32_SR_PK16_BF6_F16",
    "V_CVT_SCALEF32_SR_PK16_BF6_F32",
    "V_CVT_SCALEF32_SR_PK16_FP6_BF16",
    "V_CVT_SCALEF32_SR_PK16_FP6_F16",
    "V_CVT_SCALEF32_SR_PK16_FP6_F32",
    "V_CVT_SCALEF32_SR_PK8_BF8_BF16",
}};

}  // namespace

const BringupSummary& GetBringupSummary() {
  return kBringupSummary;
}

std::span<const std::string_view> GetFocusInstructions(BringupFocusArea area) {
  switch (area) {
    case BringupFocusArea::kVop3p:
      return kVop3pInstructions;
    case BringupFocusArea::kWmma:
      return kWmmaInstructions;
    case BringupFocusArea::kFp8Bf8:
      return kFp8Bf8Instructions;
    case BringupFocusArea::kScalePaired:
      return kScalePairedInstructions;
  }
  return std::span<const std::string_view>();
}

bool IsFocusInstruction(std::string_view instruction_name) {
  for (const BringupFocusArea area : {
           BringupFocusArea::kVop3p,
           BringupFocusArea::kWmma,
           BringupFocusArea::kFp8Bf8,
           BringupFocusArea::kScalePaired,
       }) {
    for (const std::string_view candidate : GetFocusInstructions(area)) {
      if (candidate == instruction_name) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace mirage::sim::isa::gfx1250
