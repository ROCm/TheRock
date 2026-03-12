#ifndef MIRAGE_SIM_ISA_COMMON_WAVEFRONT_SIZE_H_
#define MIRAGE_SIM_ISA_COMMON_WAVEFRONT_SIZE_H_

#include <cstdint>
#include <string_view>

namespace mirage::sim::isa {

inline constexpr std::uint32_t kWavefrontSize32 = 32;
inline constexpr std::uint32_t kWavefrontSize64 = 64;

inline constexpr std::uint32_t DefaultWavefrontSizeForGfxTarget(
    std::string_view gfx_target) {
  return (gfx_target == "gfx1201" || gfx_target == "gfx1250")
             ? kWavefrontSize32
             : kWavefrontSize64;
}

inline constexpr std::uint64_t MaskForWavefrontSize(
    std::uint32_t wavefront_size) {
  return wavefront_size >= kWavefrontSize64
             ? ~0ULL
             : ((1ULL << wavefront_size) - 1ULL);
}

}  // namespace mirage::sim::isa

#endif  // MIRAGE_SIM_ISA_COMMON_WAVEFRONT_SIZE_H_
