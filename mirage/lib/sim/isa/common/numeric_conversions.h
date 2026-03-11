#ifndef MIRAGE_SIM_ISA_COMMON_NUMERIC_CONVERSIONS_H_
#define MIRAGE_SIM_ISA_COMMON_NUMERIC_CONVERSIONS_H_

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace mirage::sim::isa {

template <typename To, typename From>
constexpr To BitCast(From value) {
  static_assert(sizeof(To) == sizeof(From));
  return std::bit_cast<To>(value);
}

inline float HalfToFloat(std::uint16_t value) {
  const std::uint32_t sign =
      static_cast<std::uint32_t>(value & 0x8000u) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1fu;
  std::uint32_t mantissa = value & 0x03ffu;

  std::uint32_t result_bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      result_bits = sign;
    } else {
      exponent = 127u - 15u + 1u;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ffu;
      result_bits = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 0x1fu) {
    result_bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    result_bits = sign | ((exponent + (127u - 15u)) << 23) |
                  (mantissa << 13);
  }
  return BitCast<float>(result_bits);
}

inline std::uint16_t FloatToHalf(float value) {
  const std::uint32_t bits = BitCast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const std::uint32_t exponent = (bits >> 23) & 0xffu;
  const std::uint32_t mantissa = bits & 0x007fffffu;

  if (exponent == 0xffu) {
    if (mantissa == 0) {
      return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    return static_cast<std::uint16_t>(sign | 0x7e00u);
  }

  const std::int32_t half_exponent =
      static_cast<std::int32_t>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }

  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }

    std::uint32_t mantissa_with_hidden = mantissa | 0x00800000u;
    const std::int32_t shift = 14 - half_exponent;
    std::uint32_t half_mantissa = mantissa_with_hidden >> shift;
    const bool round_bit =
        ((mantissa_with_hidden >> (shift - 1)) & 1u) != 0;
    const std::uint32_t sticky_mask =
        (static_cast<std::uint32_t>(1u) << (shift - 1)) - 1u;
    const bool sticky = (mantissa_with_hidden & sticky_mask) != 0;
    if (round_bit && (sticky || (half_mantissa & 1u) != 0)) {
      ++half_mantissa;
    }
    return static_cast<std::uint16_t>(sign | half_mantissa);
  }

  std::uint16_t result = static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(half_exponent) << 10) |
      (mantissa >> 13));
  const bool round_bit = ((mantissa >> 12) & 1u) != 0;
  const bool sticky = (mantissa & 0x0fffu) != 0;
  if (round_bit && (sticky || (result & 1u) != 0)) {
    ++result;
  }
  return result;
}

inline float BFloat16ToFloat(std::uint16_t value) {
  return BitCast<float>(static_cast<std::uint32_t>(value) << 16);
}

inline std::uint16_t FloatToBFloat16(float value) {
  std::uint32_t bits = BitCast<std::uint32_t>(value);
  const std::uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<std::uint16_t>(bits >> 16);
}

inline std::uint32_t PackedHalfAdd(std::uint32_t lhs, std::uint32_t rhs) {
  const std::uint16_t lhs_low = static_cast<std::uint16_t>(lhs & 0xffffu);
  const std::uint16_t lhs_high = static_cast<std::uint16_t>(lhs >> 16);
  const std::uint16_t rhs_low = static_cast<std::uint16_t>(rhs & 0xffffu);
  const std::uint16_t rhs_high = static_cast<std::uint16_t>(rhs >> 16);
  const std::uint16_t result_low = FloatToHalf(HalfToFloat(lhs_low) +
                                               HalfToFloat(rhs_low));
  const std::uint16_t result_high = FloatToHalf(HalfToFloat(lhs_high) +
                                                HalfToFloat(rhs_high));
  return static_cast<std::uint32_t>(result_low) |
         (static_cast<std::uint32_t>(result_high) << 16);
}

inline std::uint32_t PackedBFloat16Add(std::uint32_t lhs,
                                       std::uint32_t rhs) {
  const std::uint16_t lhs_low = static_cast<std::uint16_t>(lhs & 0xffffu);
  const std::uint16_t lhs_high = static_cast<std::uint16_t>(lhs >> 16);
  const std::uint16_t rhs_low = static_cast<std::uint16_t>(rhs & 0xffffu);
  const std::uint16_t rhs_high = static_cast<std::uint16_t>(rhs >> 16);
  const std::uint16_t result_low = FloatToBFloat16(BFloat16ToFloat(lhs_low) +
                                                   BFloat16ToFloat(rhs_low));
  const std::uint16_t result_high =
      FloatToBFloat16(BFloat16ToFloat(lhs_high) + BFloat16ToFloat(rhs_high));
  return static_cast<std::uint32_t>(result_low) |
         (static_cast<std::uint32_t>(result_high) << 16);
}

inline std::int16_t TruncateFloatToI16(float value) {
  if (std::isnan(value)) {
    return 0;
  }
  value = std::trunc(value);
  const float min_value =
      static_cast<float>(std::numeric_limits<std::int16_t>::min());
  const float max_value =
      static_cast<float>(std::numeric_limits<std::int16_t>::max());
  if (value <= min_value) {
    return std::numeric_limits<std::int16_t>::min();
  }
  if (value >= max_value) {
    return std::numeric_limits<std::int16_t>::max();
  }
  return static_cast<std::int16_t>(value);
}

}  // namespace mirage::sim::isa

#endif  // MIRAGE_SIM_ISA_COMMON_NUMERIC_CONVERSIONS_H_
