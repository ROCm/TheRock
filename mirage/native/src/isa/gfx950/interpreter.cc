#include "lib/sim/isa/gfx950/interpreter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "lib/sim/isa/instruction_catalog.h"

namespace mirage::sim::isa {
namespace {

template <typename To, typename From>
To BitCast(From value) {
  static_assert(sizeof(To) == sizeof(From));
  To result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

bool HasPrefix(std::string_view value, std::string_view prefix) {
  return value.size() >= prefix.size() &&
         value.substr(0, prefix.size()) == prefix;
}

bool HasSuffix(std::string_view value, std::string_view suffix) {
  return value.size() >= suffix.size() &&
         value.substr(value.size() - suffix.size()) == suffix;
}

constexpr std::uint16_t kVccPairSgprIndex = 106;
constexpr std::uint16_t kExecPairSgprIndex = 126;
constexpr std::uint16_t kSrcVcczSgprIndex = 251;
constexpr std::uint16_t kSrcExeczSgprIndex = 252;
constexpr std::uint16_t kSrcSccSgprIndex = 253;

std::uint64_t ComposeU64(std::uint32_t low, std::uint32_t high) {
  return static_cast<std::uint64_t>(low) |
         (static_cast<std::uint64_t>(high) << 32);
}

void SplitU64(std::uint64_t value,
              std::uint32_t* low,
              std::uint32_t* high) {
  if (low != nullptr) {
    *low = static_cast<std::uint32_t>(value);
  }
  if (high != nullptr) {
    *high = static_cast<std::uint32_t>(value >> 32);
  }
}

float HalfToFloat(std::uint16_t value) {
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

std::uint16_t FloatToHalf(float value) {
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

float BFloat16ToFloat(std::uint16_t value) {
  return BitCast<float>(static_cast<std::uint32_t>(value) << 16);
}

std::uint16_t FloatToBFloat16(float value) {
  std::uint32_t bits = BitCast<std::uint32_t>(value);
  const std::uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<std::uint16_t>(bits >> 16);
}

std::uint32_t ReverseBits32(std::uint32_t value);
std::uint32_t FindFirstBitHighUnsigned(std::uint32_t value);
std::uint32_t FindFirstBitLow(std::uint32_t value);
std::uint32_t FindFirstBitHighSigned(std::uint32_t value);

std::uint32_t PackedHalfAdd(std::uint32_t lhs, std::uint32_t rhs) {
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

std::uint32_t PackedBFloat16Add(std::uint32_t lhs, std::uint32_t rhs) {
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

std::uint32_t EvaluateVectorFloatBinaryF32(std::string_view opcode,
                                           std::uint32_t lhs,
                                           std::uint32_t rhs) {
  const float lhs_float = BitCast<float>(lhs);
  const float rhs_float = BitCast<float>(rhs);
  if (opcode == "V_ADD_F32") {
    return BitCast<std::uint32_t>(lhs_float + rhs_float);
  }
  if (opcode == "V_SUB_F32") {
    return BitCast<std::uint32_t>(lhs_float - rhs_float);
  }
  if (opcode == "V_MUL_F32") {
    return BitCast<std::uint32_t>(lhs_float * rhs_float);
  }
  if (opcode == "V_MIN_F32") {
    return BitCast<std::uint32_t>(std::fmin(lhs_float, rhs_float));
  }
  return BitCast<std::uint32_t>(std::fmax(lhs_float, rhs_float));
}

std::uint32_t EvaluateVectorFloatBinaryF16(std::string_view opcode,
                                           std::uint32_t lhs,
                                           std::uint32_t rhs) {
  const float lhs_float = HalfToFloat(static_cast<std::uint16_t>(lhs));
  const float rhs_float = HalfToFloat(static_cast<std::uint16_t>(rhs));
  float result = 0.0f;
  if (opcode == "V_ADD_F16") {
    result = lhs_float + rhs_float;
  } else if (opcode == "V_SUB_F16") {
    result = lhs_float - rhs_float;
  } else if (opcode == "V_MUL_F16") {
    result = lhs_float * rhs_float;
  } else if (opcode == "V_MIN_F16") {
    result = std::fmin(lhs_float, rhs_float);
  } else {
    result = std::fmax(lhs_float, rhs_float);
  }
  return static_cast<std::uint32_t>(FloatToHalf(result));
}

std::uint32_t EvaluateVectorFloatBinaryF32(CompiledOpcode opcode,
                                           std::uint32_t lhs,
                                           std::uint32_t rhs) {
  const float lhs_float = BitCast<float>(lhs);
  const float rhs_float = BitCast<float>(rhs);
  switch (opcode) {
    case CompiledOpcode::kVAddF32:
      return BitCast<std::uint32_t>(lhs_float + rhs_float);
    case CompiledOpcode::kVSubF32:
      return BitCast<std::uint32_t>(lhs_float - rhs_float);
    case CompiledOpcode::kVMulF32:
      return BitCast<std::uint32_t>(lhs_float * rhs_float);
    case CompiledOpcode::kVMinF32:
      return BitCast<std::uint32_t>(std::fmin(lhs_float, rhs_float));
    case CompiledOpcode::kVMaxF32:
      return BitCast<std::uint32_t>(std::fmax(lhs_float, rhs_float));
    default:
      return 0;
  }
}

std::uint32_t EvaluateVectorFloatBinaryF16(CompiledOpcode opcode,
                                           std::uint32_t lhs,
                                           std::uint32_t rhs) {
  const float lhs_float = HalfToFloat(static_cast<std::uint16_t>(lhs));
  const float rhs_float = HalfToFloat(static_cast<std::uint16_t>(rhs));
  float result = 0.0f;
  switch (opcode) {
    case CompiledOpcode::kVAddF16:
      result = lhs_float + rhs_float;
      break;
    case CompiledOpcode::kVSubF16:
      result = lhs_float - rhs_float;
      break;
    case CompiledOpcode::kVMulF16:
      result = lhs_float * rhs_float;
      break;
    case CompiledOpcode::kVMinF16:
      result = std::fmin(lhs_float, rhs_float);
      break;
    case CompiledOpcode::kVMaxF16:
      result = std::fmax(lhs_float, rhs_float);
      break;
    default:
      return 0;
  }
  return static_cast<std::uint32_t>(FloatToHalf(result));
}

std::uint64_t EvaluateVectorFloatBinaryF64(std::string_view opcode,
                                           std::uint64_t lhs,
                                           std::uint64_t rhs) {
  const double lhs_double = BitCast<double>(lhs);
  const double rhs_double = BitCast<double>(rhs);
  if (opcode == "V_ADD_F64") {
    return BitCast<std::uint64_t>(lhs_double + rhs_double);
  }
  if (opcode == "V_MUL_F64") {
    return BitCast<std::uint64_t>(lhs_double * rhs_double);
  }
  if (opcode == "V_MIN_F64") {
    return BitCast<std::uint64_t>(std::fmin(lhs_double, rhs_double));
  }
  return BitCast<std::uint64_t>(std::fmax(lhs_double, rhs_double));
}

std::uint64_t EvaluateVectorFloatBinaryF64(CompiledOpcode opcode,
                                           std::uint64_t lhs,
                                           std::uint64_t rhs) {
  const double lhs_double = BitCast<double>(lhs);
  const double rhs_double = BitCast<double>(rhs);
  switch (opcode) {
    case CompiledOpcode::kVAddF64:
      return BitCast<std::uint64_t>(lhs_double + rhs_double);
    case CompiledOpcode::kVMulF64:
      return BitCast<std::uint64_t>(lhs_double * rhs_double);
    case CompiledOpcode::kVMinF64:
      return BitCast<std::uint64_t>(std::fmin(lhs_double, rhs_double));
    case CompiledOpcode::kVMaxF64:
      return BitCast<std::uint64_t>(std::fmax(lhs_double, rhs_double));
    default:
      return 0u;
  }
}

std::uint32_t EvaluateVectorFloatTernaryF32(std::uint32_t src0,
                                            std::uint32_t src1,
                                            std::uint32_t src2) {
  return BitCast<std::uint32_t>(std::fma(BitCast<float>(src0),
                                         BitCast<float>(src1),
                                         BitCast<float>(src2)));
}

std::uint64_t EvaluateVectorFloatTernaryF64(std::uint64_t src0,
                                            std::uint64_t src1,
                                            std::uint64_t src2) {
  return BitCast<std::uint64_t>(std::fma(BitCast<double>(src0),
                                         BitCast<double>(src1),
                                         BitCast<double>(src2)));
}

std::int32_t TruncateFloatToI32(float value) {
  if (std::isnan(value)) {
    return 0;
  }
  const double truncated = std::trunc(static_cast<double>(value));
  if (truncated <= static_cast<double>(std::numeric_limits<std::int32_t>::min())) {
    return std::numeric_limits<std::int32_t>::min();
  }
  if (truncated >= static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
    return std::numeric_limits<std::int32_t>::max();
  }
  return static_cast<std::int32_t>(truncated);
}

std::uint32_t TruncateFloatToU32(float value) {
  if (!(value > 0.0f)) {
    return 0u;
  }
  const double truncated = std::trunc(static_cast<double>(value));
  if (!std::isfinite(truncated) ||
      truncated >= static_cast<double>(std::numeric_limits<std::uint32_t>::max())) {
    return std::numeric_limits<std::uint32_t>::max();
  }
  return static_cast<std::uint32_t>(truncated);
}

std::int16_t TruncateFloatToI16(float value) {
  if (std::isnan(value)) {
    return 0;
  }
  const double truncated = std::trunc(static_cast<double>(value));
  if (truncated <= static_cast<double>(std::numeric_limits<std::int16_t>::min())) {
    return std::numeric_limits<std::int16_t>::min();
  }
  if (truncated >= static_cast<double>(std::numeric_limits<std::int16_t>::max())) {
    return std::numeric_limits<std::int16_t>::max();
  }
  return static_cast<std::int16_t>(truncated);
}

std::uint8_t SaturateI16ToU8(std::int16_t value) {
  if (value <= 0) {
    return 0;
  }
  if (value >= 255) {
    return 255;
  }
  return static_cast<std::uint8_t>(value);
}

std::uint16_t TruncateFloatToU16(float value) {
  if (!(value > 0.0f)) {
    return 0u;
  }
  const double truncated = std::trunc(static_cast<double>(value));
  if (!std::isfinite(truncated) ||
      truncated >= static_cast<double>(std::numeric_limits<std::uint16_t>::max())) {
    return std::numeric_limits<std::uint16_t>::max();
  }
  return static_cast<std::uint16_t>(truncated);
}

std::int32_t TruncateDoubleToI32(double value) {
  if (std::isnan(value)) {
    return 0;
  }
  const double truncated = std::trunc(value);
  if (truncated <= static_cast<double>(std::numeric_limits<std::int32_t>::min())) {
    return std::numeric_limits<std::int32_t>::min();
  }
  if (truncated >= static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
    return std::numeric_limits<std::int32_t>::max();
  }
  return static_cast<std::int32_t>(truncated);
}

std::uint32_t TruncateDoubleToU32(double value) {
  if (!(value > 0.0)) {
    return 0u;
  }
  const double truncated = std::trunc(value);
  if (!std::isfinite(truncated) ||
      truncated >= static_cast<double>(std::numeric_limits<std::uint32_t>::max())) {
    return std::numeric_limits<std::uint32_t>::max();
  }
  return static_cast<std::uint32_t>(truncated);
}

std::int32_t RoundPositiveInfinityFloatToI32(float value) {
  return TruncateDoubleToI32(
      std::floor(static_cast<double>(value) + 0.5));
}

std::int32_t FloorFloatToI32(float value) {
  return TruncateDoubleToI32(std::floor(static_cast<double>(value)));
}

float EvaluateUnaryFloatMathF16(std::string_view opcode, float input) {
  if (opcode == "V_RCP_F16") {
    return 1.0f / input;
  }
  if (opcode == "V_SQRT_F16") {
    return std::sqrt(input);
  }
  if (opcode == "V_RSQ_F16") {
    return 1.0f / std::sqrt(input);
  }
  if (opcode == "V_LOG_F16") {
    return std::log2(input);
  }
  if (opcode == "V_EXP_F16") {
    return std::exp2(input);
  }
  if (opcode == "V_SIN_F16") {
    return std::sin(input);
  }
  return std::cos(input);
}

float EvaluateUnaryFloatMathF16(CompiledOpcode opcode, float input) {
  switch (opcode) {
    case CompiledOpcode::kVRcpF16:
      return 1.0f / input;
    case CompiledOpcode::kVSqrtF16:
      return std::sqrt(input);
    case CompiledOpcode::kVRsqF16:
      return 1.0f / std::sqrt(input);
    case CompiledOpcode::kVLogF16:
      return std::log2(input);
    case CompiledOpcode::kVExpF16:
      return std::exp2(input);
    case CompiledOpcode::kVSinF16:
      return std::sin(input);
    case CompiledOpcode::kVCosF16:
      return std::cos(input);
    default:
      return 0.0f;
  }
}

float EvaluateUnaryFloatMathF32(std::string_view opcode, float input) {
  if (opcode == "V_EXP_F32" || opcode == "V_EXP_LEGACY_F32") {
    return std::exp2(input);
  }
  if (opcode == "V_LOG_F32" || opcode == "V_LOG_LEGACY_F32") {
    return std::log2(input);
  }
  if (opcode == "V_RCP_F32" || opcode == "V_RCP_IFLAG_F32") {
    return 1.0f / input;
  }
  if (opcode == "V_RSQ_F32") {
    return 1.0f / std::sqrt(input);
  }
  if (opcode == "V_SQRT_F32") {
    return std::sqrt(input);
  }
  if (opcode == "V_SIN_F32") {
    return std::sin(input);
  }
  return std::cos(input);
}

float EvaluateUnaryFloatMathF32(CompiledOpcode opcode, float input) {
  switch (opcode) {
    case CompiledOpcode::kVExpF32:
    case CompiledOpcode::kVExpLegacyF32:
      return std::exp2(input);
    case CompiledOpcode::kVLogF32:
    case CompiledOpcode::kVLogLegacyF32:
      return std::log2(input);
    case CompiledOpcode::kVRcpF32:
    case CompiledOpcode::kVRcpIflagF32:
      return 1.0f / input;
    case CompiledOpcode::kVRsqF32:
      return 1.0f / std::sqrt(input);
    case CompiledOpcode::kVSqrtF32:
      return std::sqrt(input);
    case CompiledOpcode::kVSinF32:
      return std::sin(input);
    case CompiledOpcode::kVCosF32:
      return std::cos(input);
    default:
      return 0.0f;
  }
}

double EvaluateUnaryFloatMathF64(std::string_view opcode, double input) {
  if (opcode == "V_RCP_F64") {
    return 1.0 / input;
  }
  if (opcode == "V_RSQ_F64") {
    return 1.0 / std::sqrt(input);
  }
  return std::sqrt(input);
}

double EvaluateUnaryFloatMathF64(CompiledOpcode opcode, double input) {
  switch (opcode) {
    case CompiledOpcode::kVRcpF64:
      return 1.0 / input;
    case CompiledOpcode::kVRsqF64:
      return 1.0 / std::sqrt(input);
    case CompiledOpcode::kVSqrtF64:
      return std::sqrt(input);
    default:
      return 0.0;
  }
}

std::int32_t EvaluateFrexpExpI32(float input) {
  if (input == 0.0f || !std::isfinite(input)) {
    return 0;
  }
  int exponent = 0;
  std::frexp(input, &exponent);
  return exponent;
}

std::int32_t EvaluateFrexpExpI32(double input) {
  if (input == 0.0 || !std::isfinite(input)) {
    return 0;
  }
  int exponent = 0;
  std::frexp(input, &exponent);
  return exponent;
}

std::int16_t EvaluateFrexpExpI16(float input) {
  const std::int32_t exponent = EvaluateFrexpExpI32(input);
  if (exponent <= static_cast<std::int32_t>(
                      std::numeric_limits<std::int16_t>::min())) {
    return std::numeric_limits<std::int16_t>::min();
  }
  if (exponent >= static_cast<std::int32_t>(
                      std::numeric_limits<std::int16_t>::max())) {
    return std::numeric_limits<std::int16_t>::max();
  }
  return static_cast<std::int16_t>(exponent);
}

float EvaluateFrexpMantissaF32(float input) {
  if (input == 0.0f || !std::isfinite(input)) {
    return input;
  }
  int exponent = 0;
  return std::frexp(input, &exponent);
}

double EvaluateFrexpMantissaF64(double input) {
  if (input == 0.0 || !std::isfinite(input)) {
    return input;
  }
  int exponent = 0;
  return std::frexp(input, &exponent);
}

std::uint32_t EvaluateVectorUnary32(std::string_view opcode, std::uint32_t value) {
  if (opcode == "V_NOT_B32") {
    return ~value;
  }
  if (opcode == "V_BFREV_B32") {
    return ReverseBits32(value);
  }
  if (opcode == "V_FFBH_U32") {
    return FindFirstBitHighUnsigned(value);
  }
  if (opcode == "V_FFBL_B32") {
    return FindFirstBitLow(value);
  }
  if (opcode == "V_FFBH_I32") {
    return FindFirstBitHighSigned(value);
  }
  if (opcode == "V_CVT_F16_U16") {
    return static_cast<std::uint32_t>(
        FloatToHalf(static_cast<float>(value & 0xffffu)));
  }
  if (opcode == "V_CVT_F16_I16") {
    return static_cast<std::uint32_t>(FloatToHalf(
        static_cast<float>(static_cast<std::int16_t>(value & 0xffffu))));
  }
  if (opcode == "V_CVT_U16_F16") {
    return static_cast<std::uint32_t>(TruncateFloatToU16(
        HalfToFloat(static_cast<std::uint16_t>(value))));
  }
  if (opcode == "V_CVT_I16_F16") {
    return static_cast<std::uint32_t>(static_cast<std::uint16_t>(
        TruncateFloatToI16(HalfToFloat(static_cast<std::uint16_t>(value)))));
  }
  if (opcode == "V_SAT_PK_U8_I16") {
    const std::uint8_t low = SaturateI16ToU8(
        static_cast<std::int16_t>(value & 0xffffu));
    const std::uint8_t high = SaturateI16ToU8(
        static_cast<std::int16_t>((value >> 16) & 0xffffu));
    return static_cast<std::uint32_t>(low) |
           (static_cast<std::uint32_t>(high) << 8);
  }
  if (opcode == "V_CVT_F32_UBYTE0") {
    return BitCast<std::uint32_t>(
        static_cast<float>(static_cast<std::uint8_t>(value & 0xffu)));
  }
  if (opcode == "V_CVT_F32_UBYTE1") {
    return BitCast<std::uint32_t>(
        static_cast<float>(static_cast<std::uint8_t>((value >> 8) & 0xffu)));
  }
  if (opcode == "V_CVT_F32_UBYTE2") {
    return BitCast<std::uint32_t>(
        static_cast<float>(static_cast<std::uint8_t>((value >> 16) & 0xffu)));
  }
  if (opcode == "V_CVT_F32_UBYTE3") {
    return BitCast<std::uint32_t>(
        static_cast<float>(static_cast<std::uint8_t>((value >> 24) & 0xffu)));
  }
  if (opcode == "V_CVT_F32_I32") {
    return BitCast<std::uint32_t>(static_cast<float>(BitCast<std::int32_t>(value)));
  }
  if (opcode == "V_CVT_F32_U32") {
    return BitCast<std::uint32_t>(static_cast<float>(value));
  }
  if (opcode == "V_CVT_U32_F32") {
    return TruncateFloatToU32(BitCast<float>(value));
  }
  if (opcode == "V_CVT_I32_F32") {
    return BitCast<std::uint32_t>(TruncateFloatToI32(BitCast<float>(value)));
  }
  if (opcode == "V_CVT_RPI_I32_F32") {
    return BitCast<std::uint32_t>(
        RoundPositiveInfinityFloatToI32(BitCast<float>(value)));
  }
  if (opcode == "V_CVT_FLR_I32_F32") {
    return BitCast<std::uint32_t>(FloorFloatToI32(BitCast<float>(value)));
  }
  if (opcode == "V_CVT_F16_F32") {
    return static_cast<std::uint32_t>(FloatToHalf(BitCast<float>(value)));
  }
  if (opcode == "V_CVT_F32_F16") {
    return BitCast<std::uint32_t>(HalfToFloat(static_cast<std::uint16_t>(value)));
  }
  if (opcode == "V_RCP_F32" || opcode == "V_RCP_IFLAG_F32" ||
      opcode == "V_RSQ_F32" || opcode == "V_SQRT_F32" ||
      opcode == "V_LOG_F32" || opcode == "V_LOG_LEGACY_F32" ||
      opcode == "V_EXP_F32" || opcode == "V_EXP_LEGACY_F32" ||
      opcode == "V_SIN_F32" || opcode == "V_COS_F32") {
    return BitCast<std::uint32_t>(
        EvaluateUnaryFloatMathF32(opcode, BitCast<float>(value)));
  }
  if (opcode == "V_FREXP_EXP_I32_F32") {
    return BitCast<std::uint32_t>(EvaluateFrexpExpI32(BitCast<float>(value)));
  }
  if (opcode == "V_FREXP_MANT_F32") {
    return BitCast<std::uint32_t>(
        EvaluateFrexpMantissaF32(BitCast<float>(value)));
  }
  if (opcode == "V_FRACT_F32") {
    const float input = BitCast<float>(value);
    return BitCast<std::uint32_t>(input - std::floor(input));
  }
  if (opcode == "V_TRUNC_F32") {
    return BitCast<std::uint32_t>(std::trunc(BitCast<float>(value)));
  }
  if (opcode == "V_CEIL_F32") {
    return BitCast<std::uint32_t>(std::ceil(BitCast<float>(value)));
  }
  if (opcode == "V_RNDNE_F32") {
    return BitCast<std::uint32_t>(std::nearbyint(BitCast<float>(value)));
  }
  if (opcode == "V_FLOOR_F32") {
    return BitCast<std::uint32_t>(std::floor(BitCast<float>(value)));
  }
  if (opcode == "V_RCP_F16" || opcode == "V_SQRT_F16" ||
      opcode == "V_RSQ_F16" || opcode == "V_LOG_F16" ||
      opcode == "V_EXP_F16" || opcode == "V_SIN_F16" ||
      opcode == "V_COS_F16") {
    return static_cast<std::uint32_t>(FloatToHalf(EvaluateUnaryFloatMathF16(
        opcode, HalfToFloat(static_cast<std::uint16_t>(value)))));
  }
  if (opcode == "V_FREXP_EXP_I16_F16") {
    return static_cast<std::uint32_t>(static_cast<std::uint16_t>(
        EvaluateFrexpExpI16(HalfToFloat(static_cast<std::uint16_t>(value)))));
  }
  if (opcode == "V_FREXP_MANT_F16") {
    return static_cast<std::uint32_t>(FloatToHalf(EvaluateFrexpMantissaF32(
        HalfToFloat(static_cast<std::uint16_t>(value)))));
  }
  if (opcode == "V_FRACT_F16") {
    const float input = HalfToFloat(static_cast<std::uint16_t>(value));
    return static_cast<std::uint32_t>(FloatToHalf(input - std::floor(input)));
  }
  if (opcode == "V_TRUNC_F16") {
    return static_cast<std::uint32_t>(
        FloatToHalf(std::trunc(HalfToFloat(static_cast<std::uint16_t>(value)))));
  }
  if (opcode == "V_CEIL_F16") {
    return static_cast<std::uint32_t>(
        FloatToHalf(std::ceil(HalfToFloat(static_cast<std::uint16_t>(value)))));
  }
  if (opcode == "V_RNDNE_F16") {
    return static_cast<std::uint32_t>(
        FloatToHalf(std::nearbyint(HalfToFloat(static_cast<std::uint16_t>(value)))));
  }
  if (opcode == "V_FLOOR_F16") {
    return static_cast<std::uint32_t>(
        FloatToHalf(std::floor(HalfToFloat(static_cast<std::uint16_t>(value)))));
  }
  return 0u;
}

std::uint32_t EvaluateVectorUnary32(CompiledOpcode opcode, std::uint32_t value) {
  switch (opcode) {
    case CompiledOpcode::kVNotB32:
      return ~value;
    case CompiledOpcode::kVBfrevB32:
      return ReverseBits32(value);
    case CompiledOpcode::kVFfbhU32:
      return FindFirstBitHighUnsigned(value);
    case CompiledOpcode::kVFfblB32:
      return FindFirstBitLow(value);
    case CompiledOpcode::kVFfbhI32:
      return FindFirstBitHighSigned(value);
    case CompiledOpcode::kVCvtF16U16:
      return static_cast<std::uint32_t>(
          FloatToHalf(static_cast<float>(value & 0xffffu)));
    case CompiledOpcode::kVCvtF16I16:
      return static_cast<std::uint32_t>(FloatToHalf(
          static_cast<float>(static_cast<std::int16_t>(value & 0xffffu))));
    case CompiledOpcode::kVCvtU16F16:
      return static_cast<std::uint32_t>(TruncateFloatToU16(
          HalfToFloat(static_cast<std::uint16_t>(value))));
    case CompiledOpcode::kVCvtI16F16:
      return static_cast<std::uint32_t>(static_cast<std::uint16_t>(
          TruncateFloatToI16(HalfToFloat(static_cast<std::uint16_t>(value)))));
    case CompiledOpcode::kVSatPkU8I16: {
      const std::uint8_t low = SaturateI16ToU8(
          static_cast<std::int16_t>(value & 0xffffu));
      const std::uint8_t high = SaturateI16ToU8(
          static_cast<std::int16_t>((value >> 16) & 0xffffu));
      return static_cast<std::uint32_t>(low) |
             (static_cast<std::uint32_t>(high) << 8);
    }
    case CompiledOpcode::kVCvtF32Ubyte0:
      return BitCast<std::uint32_t>(
          static_cast<float>(static_cast<std::uint8_t>(value & 0xffu)));
    case CompiledOpcode::kVCvtF32Ubyte1:
      return BitCast<std::uint32_t>(
          static_cast<float>(static_cast<std::uint8_t>((value >> 8) & 0xffu)));
    case CompiledOpcode::kVCvtF32Ubyte2:
      return BitCast<std::uint32_t>(
          static_cast<float>(static_cast<std::uint8_t>((value >> 16) & 0xffu)));
    case CompiledOpcode::kVCvtF32Ubyte3:
      return BitCast<std::uint32_t>(
          static_cast<float>(static_cast<std::uint8_t>((value >> 24) & 0xffu)));
    case CompiledOpcode::kVCvtF32I32:
      return BitCast<std::uint32_t>(
          static_cast<float>(BitCast<std::int32_t>(value)));
    case CompiledOpcode::kVCvtF32U32:
      return BitCast<std::uint32_t>(static_cast<float>(value));
    case CompiledOpcode::kVCvtU32F32:
      return TruncateFloatToU32(BitCast<float>(value));
    case CompiledOpcode::kVCvtI32F32:
      return BitCast<std::uint32_t>(TruncateFloatToI32(BitCast<float>(value)));
    case CompiledOpcode::kVCvtRpiI32F32:
      return BitCast<std::uint32_t>(
          RoundPositiveInfinityFloatToI32(BitCast<float>(value)));
    case CompiledOpcode::kVCvtFlrI32F32:
      return BitCast<std::uint32_t>(FloorFloatToI32(BitCast<float>(value)));
    case CompiledOpcode::kVCvtF16F32:
      return static_cast<std::uint32_t>(FloatToHalf(BitCast<float>(value)));
    case CompiledOpcode::kVCvtF32F16:
      return BitCast<std::uint32_t>(
          HalfToFloat(static_cast<std::uint16_t>(value)));
    case CompiledOpcode::kVRcpF32:
    case CompiledOpcode::kVRcpIflagF32:
    case CompiledOpcode::kVRsqF32:
    case CompiledOpcode::kVSqrtF32:
    case CompiledOpcode::kVLogF32:
    case CompiledOpcode::kVLogLegacyF32:
    case CompiledOpcode::kVExpF32:
    case CompiledOpcode::kVExpLegacyF32:
    case CompiledOpcode::kVSinF32:
    case CompiledOpcode::kVCosF32:
      return BitCast<std::uint32_t>(
          EvaluateUnaryFloatMathF32(opcode, BitCast<float>(value)));
    case CompiledOpcode::kVFrexpExpI32F32:
      return BitCast<std::uint32_t>(EvaluateFrexpExpI32(BitCast<float>(value)));
    case CompiledOpcode::kVFrexpMantF32:
      return BitCast<std::uint32_t>(
          EvaluateFrexpMantissaF32(BitCast<float>(value)));
    case CompiledOpcode::kVFractF32: {
      const float input = BitCast<float>(value);
      return BitCast<std::uint32_t>(input - std::floor(input));
    }
    case CompiledOpcode::kVTruncF32:
      return BitCast<std::uint32_t>(std::trunc(BitCast<float>(value)));
    case CompiledOpcode::kVCeilF32:
      return BitCast<std::uint32_t>(std::ceil(BitCast<float>(value)));
    case CompiledOpcode::kVRndneF32:
      return BitCast<std::uint32_t>(std::nearbyint(BitCast<float>(value)));
    case CompiledOpcode::kVFloorF32:
      return BitCast<std::uint32_t>(std::floor(BitCast<float>(value)));
    case CompiledOpcode::kVRcpF16:
    case CompiledOpcode::kVSqrtF16:
    case CompiledOpcode::kVRsqF16:
    case CompiledOpcode::kVLogF16:
    case CompiledOpcode::kVExpF16:
    case CompiledOpcode::kVSinF16:
    case CompiledOpcode::kVCosF16:
      return static_cast<std::uint32_t>(FloatToHalf(
          EvaluateUnaryFloatMathF16(
              opcode, HalfToFloat(static_cast<std::uint16_t>(value)))));
    case CompiledOpcode::kVFrexpExpI16F16:
      return static_cast<std::uint32_t>(static_cast<std::uint16_t>(
          EvaluateFrexpExpI16(HalfToFloat(static_cast<std::uint16_t>(value)))));
    case CompiledOpcode::kVFrexpMantF16:
      return static_cast<std::uint32_t>(FloatToHalf(EvaluateFrexpMantissaF32(
          HalfToFloat(static_cast<std::uint16_t>(value)))));
    case CompiledOpcode::kVFractF16: {
      const float input = HalfToFloat(static_cast<std::uint16_t>(value));
      return static_cast<std::uint32_t>(FloatToHalf(input - std::floor(input)));
    }
    case CompiledOpcode::kVTruncF16:
      return static_cast<std::uint32_t>(
          FloatToHalf(std::trunc(HalfToFloat(static_cast<std::uint16_t>(value)))));
    case CompiledOpcode::kVCeilF16:
      return static_cast<std::uint32_t>(
          FloatToHalf(std::ceil(HalfToFloat(static_cast<std::uint16_t>(value)))));
    case CompiledOpcode::kVRndneF16:
      return static_cast<std::uint32_t>(FloatToHalf(
          std::nearbyint(HalfToFloat(static_cast<std::uint16_t>(value)))));
    case CompiledOpcode::kVFloorF16:
      return static_cast<std::uint32_t>(
          FloatToHalf(std::floor(HalfToFloat(static_cast<std::uint16_t>(value)))));
    default:
      return 0u;
  }
}

std::uint64_t EvaluateVectorUnary64(std::string_view opcode, std::uint32_t value) {
  if (opcode == "V_CVT_F64_F32") {
    return BitCast<std::uint64_t>(static_cast<double>(BitCast<float>(value)));
  }
  if (opcode == "V_CVT_F64_I32") {
    return BitCast<std::uint64_t>(
        static_cast<double>(BitCast<std::int32_t>(value)));
  }
  if (opcode == "V_CVT_F64_U32") {
    return BitCast<std::uint64_t>(static_cast<double>(value));
  }
  return 0u;
}

std::uint64_t EvaluateVectorUnary64(CompiledOpcode opcode, std::uint32_t value) {
  switch (opcode) {
    case CompiledOpcode::kVCvtF64F32:
      return BitCast<std::uint64_t>(static_cast<double>(BitCast<float>(value)));
    case CompiledOpcode::kVCvtF64I32:
      return BitCast<std::uint64_t>(
          static_cast<double>(BitCast<std::int32_t>(value)));
    case CompiledOpcode::kVCvtF64U32:
      return BitCast<std::uint64_t>(static_cast<double>(value));
    default:
      return 0u;
  }
}

std::uint32_t EvaluateVectorUnaryFrom64(std::string_view opcode, std::uint64_t value) {
  if (opcode == "V_CVT_F32_F64") {
    return BitCast<std::uint32_t>(static_cast<float>(BitCast<double>(value)));
  }
  if (opcode == "V_CVT_I32_F64") {
    return BitCast<std::uint32_t>(TruncateDoubleToI32(BitCast<double>(value)));
  }
  if (opcode == "V_CVT_U32_F64") {
    return TruncateDoubleToU32(BitCast<double>(value));
  }
  if (opcode == "V_FREXP_EXP_I32_F64") {
    return BitCast<std::uint32_t>(EvaluateFrexpExpI32(BitCast<double>(value)));
  }
  return 0u;
}

std::uint32_t EvaluateVectorUnaryFrom64(CompiledOpcode opcode, std::uint64_t value) {
  switch (opcode) {
    case CompiledOpcode::kVCvtF32F64:
      return BitCast<std::uint32_t>(static_cast<float>(BitCast<double>(value)));
    case CompiledOpcode::kVCvtI32F64:
      return BitCast<std::uint32_t>(TruncateDoubleToI32(BitCast<double>(value)));
    case CompiledOpcode::kVCvtU32F64:
      return TruncateDoubleToU32(BitCast<double>(value));
    case CompiledOpcode::kVFrexpExpI32F64:
      return BitCast<std::uint32_t>(EvaluateFrexpExpI32(BitCast<double>(value)));
    default:
      return 0u;
  }
}

std::uint64_t EvaluateVectorUnary64To64(std::string_view opcode,
                                        std::uint64_t value) {
  if (opcode == "V_RCP_F64" || opcode == "V_RSQ_F64" ||
      opcode == "V_SQRT_F64") {
    return BitCast<std::uint64_t>(
        EvaluateUnaryFloatMathF64(opcode, BitCast<double>(value)));
  }
  if (opcode == "V_FREXP_MANT_F64") {
    return BitCast<std::uint64_t>(
        EvaluateFrexpMantissaF64(BitCast<double>(value)));
  }
  if (opcode == "V_FRACT_F64") {
    const double input = BitCast<double>(value);
    return BitCast<std::uint64_t>(input - std::floor(input));
  }
  if (opcode == "V_TRUNC_F64") {
    return BitCast<std::uint64_t>(std::trunc(BitCast<double>(value)));
  }
  if (opcode == "V_CEIL_F64") {
    return BitCast<std::uint64_t>(std::ceil(BitCast<double>(value)));
  }
  if (opcode == "V_RNDNE_F64") {
    return BitCast<std::uint64_t>(std::nearbyint(BitCast<double>(value)));
  }
  if (opcode == "V_FLOOR_F64") {
    return BitCast<std::uint64_t>(std::floor(BitCast<double>(value)));
  }
  return 0u;
}

std::uint64_t EvaluateVectorUnary64To64(CompiledOpcode opcode,
                                        std::uint64_t value) {
  switch (opcode) {
    case CompiledOpcode::kVRcpF64:
    case CompiledOpcode::kVRsqF64:
    case CompiledOpcode::kVSqrtF64:
      return BitCast<std::uint64_t>(
          EvaluateUnaryFloatMathF64(opcode, BitCast<double>(value)));
    case CompiledOpcode::kVFrexpMantF64:
      return BitCast<std::uint64_t>(
          EvaluateFrexpMantissaF64(BitCast<double>(value)));
    case CompiledOpcode::kVFractF64: {
      const double input = BitCast<double>(value);
      return BitCast<std::uint64_t>(input - std::floor(input));
    }
    case CompiledOpcode::kVTruncF64:
      return BitCast<std::uint64_t>(std::trunc(BitCast<double>(value)));
    case CompiledOpcode::kVCeilF64:
      return BitCast<std::uint64_t>(std::ceil(BitCast<double>(value)));
    case CompiledOpcode::kVRndneF64:
      return BitCast<std::uint64_t>(std::nearbyint(BitCast<double>(value)));
    case CompiledOpcode::kVFloorF64:
      return BitCast<std::uint64_t>(std::floor(BitCast<double>(value)));
    default:
      return 0u;
  }
}

std::uint64_t ReverseBits64(std::uint64_t value) {
  value = ((value & 0x5555555555555555ULL) << 1) |
          ((value >> 1) & 0x5555555555555555ULL);
  value = ((value & 0x3333333333333333ULL) << 2) |
          ((value >> 2) & 0x3333333333333333ULL);
  value = ((value & 0x0f0f0f0f0f0f0f0fULL) << 4) |
          ((value >> 4) & 0x0f0f0f0f0f0f0f0fULL);
  value = ((value & 0x00ff00ff00ff00ffULL) << 8) |
          ((value >> 8) & 0x00ff00ff00ff00ffULL);
  value = ((value & 0x0000ffff0000ffffULL) << 16) |
          ((value >> 16) & 0x0000ffff0000ffffULL);
  return (value << 32) | (value >> 32);
}

std::uint32_t PopCount32(std::uint32_t value) {
#if defined(__GNUC__) || defined(__clang__)
  return static_cast<std::uint32_t>(__builtin_popcount(value));
#else
  std::uint32_t count = 0;
  while (value != 0u) {
    value &= value - 1u;
    ++count;
  }
  return count;
#endif
}

std::uint32_t PopCount64(std::uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
  return static_cast<std::uint32_t>(__builtin_popcountll(value));
#else
  std::uint32_t count = 0;
  while (value != 0u) {
    value &= value - 1u;
    ++count;
  }
  return count;
#endif
}

std::uint8_t GetGlobalAtomicMemoryDwordCount(std::string_view opcode) {
  if (HasSuffix(opcode, "_X2") || opcode == "GLOBAL_ATOMIC_ADD_F64" ||
      opcode == "GLOBAL_ATOMIC_MIN_F64" ||
      opcode == "GLOBAL_ATOMIC_MAX_F64") {
    return 2;
  }
  return 1;
}

bool IsScalarMoveOpcode(std::string_view opcode) {
  return opcode == "S_MOV_B32" || opcode == "S_MOVK_I32" ||
         opcode == "S_CMOVK_I32" ||
         opcode == "S_MOV_B64" || opcode == "S_CMOV_B32" ||
         opcode == "S_CMOV_B64" || opcode == "S_NOT_B32" ||
         opcode == "S_NOT_B64" || opcode == "S_BREV_B32" ||
         opcode == "S_BREV_B64" ||
         opcode == "S_BCNT0_I32_B32" || opcode == "S_BCNT0_I32_B64" ||
         opcode == "S_BCNT1_I32_B32" || opcode == "S_BCNT1_I32_B64" ||
         opcode == "S_FF0_I32_B32" || opcode == "S_FF0_I32_B64" ||
         opcode == "S_FF1_I32_B32" || opcode == "S_FF1_I32_B64" ||
         opcode == "S_FLBIT_I32_B32" || opcode == "S_FLBIT_I32_B64" ||
         opcode == "S_FLBIT_I32" || opcode == "S_FLBIT_I32_I64" ||
         opcode == "S_BITREPLICATE_B64_B32" ||
         opcode == "S_QUADMASK_B32" || opcode == "S_QUADMASK_B64" ||
         opcode == "S_ABS_I32" ||
         opcode == "S_SEXT_I32_I8" ||
         opcode == "S_SEXT_I32_I16" || opcode == "S_BITSET0_B32" ||
         opcode == "S_BITSET0_B64" || opcode == "S_BITSET1_B32" ||
         opcode == "S_BITSET1_B64";
}

bool IsScalarBinaryOpcode(std::string_view opcode) {
  return opcode == "S_ADD_U32" || opcode == "S_ADD_I32" ||
         opcode == "S_ADDK_I32" ||
         opcode == "S_ADDC_U32" || opcode == "S_SUB_U32" ||
         opcode == "S_SUB_I32" ||
         opcode == "S_SUBB_U32" || opcode == "S_MUL_I32" ||
         opcode == "S_MULK_I32" || opcode == "S_MUL_HI_U32" ||
         opcode == "S_MUL_HI_I32" || opcode == "S_LSHL_B32" ||
         opcode == "S_LSHL_B64" || opcode == "S_LSHR_B32" ||
         opcode == "S_LSHR_B64" || opcode == "S_ASHR_I32" ||
         opcode == "S_ASHR_I64" || opcode == "S_BFM_B32" ||
         opcode == "S_BFM_B64" || opcode == "S_LSHL1_ADD_U32" ||
         opcode == "S_LSHL2_ADD_U32" || opcode == "S_LSHL3_ADD_U32" ||
         opcode == "S_LSHL4_ADD_U32" || opcode == "S_PACK_LL_B32_B16" ||
         opcode == "S_PACK_LH_B32_B16" ||
         opcode == "S_PACK_HH_B32_B16" ||
         opcode == "S_MIN_I32" || opcode == "S_MIN_U32" ||
         opcode == "S_MAX_I32" || opcode == "S_MAX_U32" ||
         opcode == "S_CSELECT_B32" || opcode == "S_CSELECT_B64" ||
         opcode == "S_ABSDIFF_I32" ||
         opcode == "S_BFE_U32" || opcode == "S_BFE_I32" ||
         opcode == "S_BFE_U64" || opcode == "S_BFE_I64" ||
         opcode == "S_AND_B32" || opcode == "S_ANDN2_B32" ||
         opcode == "S_NAND_B32" || opcode == "S_OR_B32" ||
         opcode == "S_ORN2_B32" || opcode == "S_NOR_B32" ||
         opcode == "S_XOR_B32" || opcode == "S_XNOR_B32" ||
         opcode == "S_AND_B64" ||
         opcode == "S_ANDN2_B64" || opcode == "S_NAND_B64" ||
         opcode == "S_OR_B64" || opcode == "S_ORN2_B64" ||
         opcode == "S_NOR_B64" || opcode == "S_XOR_B64" ||
         opcode == "S_XNOR_B64";
}

bool IsExecMaskOpcode(std::string_view opcode) {
  return opcode == "S_AND_SAVEEXEC_B64" ||
         opcode == "S_ANDN1_SAVEEXEC_B64" ||
         opcode == "S_ANDN2_SAVEEXEC_B64" ||
         opcode == "S_NAND_SAVEEXEC_B64" ||
         opcode == "S_OR_SAVEEXEC_B64" ||
         opcode == "S_ORN1_SAVEEXEC_B64" ||
         opcode == "S_ORN2_SAVEEXEC_B64" ||
         opcode == "S_NOR_SAVEEXEC_B64" ||
         opcode == "S_XOR_SAVEEXEC_B64" ||
         opcode == "S_XNOR_SAVEEXEC_B64" ||
         opcode == "S_ANDN1_WREXEC_B64" ||
         opcode == "S_ANDN2_WREXEC_B64";
}

bool IsWrExecOpcode(std::string_view opcode) {
  return opcode == "S_ANDN1_WREXEC_B64" || opcode == "S_ANDN2_WREXEC_B64";
}

bool IsWrExecOpcode(CompiledOpcode opcode) {
  return opcode == CompiledOpcode::kSAndn1WrexecB64 ||
         opcode == CompiledOpcode::kSAndn2WrexecB64;
}

bool IsScalarCompareOpcode(std::string_view opcode) {
  return opcode == "S_CMP_EQ_I32" || opcode == "S_CMPK_EQ_I32" ||
         opcode == "S_CMP_LG_I32" || opcode == "S_CMPK_LG_I32" ||
         opcode == "S_CMP_GT_I32" || opcode == "S_CMPK_GT_I32" ||
         opcode == "S_CMP_GE_I32" || opcode == "S_CMPK_GE_I32" ||
         opcode == "S_CMP_LT_I32" || opcode == "S_CMPK_LT_I32" ||
         opcode == "S_CMP_LE_I32" || opcode == "S_CMPK_LE_I32" ||
         opcode == "S_CMP_EQ_U32" || opcode == "S_CMPK_EQ_U32" ||
         opcode == "S_CMP_LG_U32" || opcode == "S_CMPK_LG_U32" ||
         opcode == "S_CMP_GT_U32" || opcode == "S_CMPK_GT_U32" ||
         opcode == "S_CMP_GE_U32" || opcode == "S_CMPK_GE_U32" ||
         opcode == "S_CMP_LT_U32" || opcode == "S_CMPK_LT_U32" ||
         opcode == "S_CMP_LE_U32" || opcode == "S_CMPK_LE_U32" ||
         opcode == "S_CMP_EQ_U64" || opcode == "S_CMP_LG_U64" ||
         opcode == "S_BITCMP0_B32" || opcode == "S_BITCMP1_B32" ||
         opcode == "S_BITCMP0_B64" || opcode == "S_BITCMP1_B64";
}

bool IsScalarMemoryOpcode(std::string_view opcode) {
  return opcode == "S_LOAD_DWORD" || opcode == "S_LOAD_DWORDX2" ||
         opcode == "S_STORE_DWORD";
}

bool IsVectorBinaryOpcode(std::string_view opcode) {
  return opcode == "V_CNDMASK_B32" ||
         opcode == "V_ADD_U32" || opcode == "V_ADD_CO_U32" ||
         opcode == "V_ADDC_CO_U32" ||
         opcode == "V_ADD_F16" || opcode == "V_SUB_F16" ||
         opcode == "V_MUL_F16" || opcode == "V_MIN_F16" ||
         opcode == "V_MAX_F16" ||
         opcode == "V_ADD_F32" || opcode == "V_SUB_F32" ||
         opcode == "V_MUL_F32" || opcode == "V_MIN_F32" ||
         opcode == "V_MAX_F32" ||
         opcode == "V_SUB_U32" || opcode == "V_SUB_CO_U32" ||
         opcode == "V_SUBB_CO_U32" ||
         opcode == "V_SUBREV_U32" || opcode == "V_SUBREV_CO_U32" ||
         opcode == "V_SUBBREV_CO_U32" ||
         opcode == "V_MUL_LO_U32" ||
         opcode == "V_MUL_HI_U32" ||
         opcode == "V_MUL_HI_I32" || opcode == "V_BCNT_U32_B32" ||
         opcode == "V_BFM_B32" ||
         opcode == "V_MBCNT_LO_U32_B32" || opcode == "V_MBCNT_HI_U32_B32" ||
         opcode == "V_LSHLREV_B64" || opcode == "V_LSHRREV_B64" ||
         opcode == "V_ASHRREV_I64" ||
         opcode == "V_ADD_F64" || opcode == "V_MUL_F64" ||
         opcode == "V_MIN_F64" || opcode == "V_MAX_F64" ||
         opcode == "V_MIN_I32" || opcode == "V_MAX_I32" ||
         opcode == "V_MIN_U32" || opcode == "V_MAX_U32" ||
         opcode == "V_LSHRREV_B32" || opcode == "V_ASHRREV_I32" ||
         opcode == "V_LSHLREV_B32" || opcode == "V_WRITELANE_B32" ||
         opcode == "V_AND_B32" ||
         opcode == "V_OR_B32" || opcode == "V_XOR_B32";
}

bool IsVectorCarryOutBinaryOpcode(std::string_view opcode) {
  return opcode == "V_ADD_CO_U32" || opcode == "V_SUB_CO_U32" ||
         opcode == "V_SUBREV_CO_U32";
}

bool IsVectorCarryInBinaryOpcode(std::string_view opcode) {
  return opcode == "V_ADDC_CO_U32" || opcode == "V_SUBB_CO_U32" ||
         opcode == "V_SUBBREV_CO_U32";
}

bool IsVectorUnaryOpcode(std::string_view opcode) {
  return opcode == "V_NOT_B32" || opcode == "V_BFREV_B32" ||
         opcode == "V_FFBH_U32" || opcode == "V_FFBL_B32" ||
         opcode == "V_FFBH_I32" ||
         opcode == "V_CVT_F16_U16" ||
         opcode == "V_CVT_F16_I16" ||
         opcode == "V_CVT_U16_F16" ||
         opcode == "V_CVT_I16_F16" ||
         opcode == "V_SAT_PK_U8_I16" ||
         opcode == "V_CVT_F32_UBYTE0" ||
         opcode == "V_CVT_F32_UBYTE1" ||
         opcode == "V_CVT_F32_UBYTE2" ||
         opcode == "V_CVT_F32_UBYTE3" ||
         opcode == "V_RCP_F16" || opcode == "V_SQRT_F16" ||
         opcode == "V_RSQ_F16" || opcode == "V_LOG_F16" ||
         opcode == "V_EXP_F16" || opcode == "V_SIN_F16" ||
         opcode == "V_COS_F16" || opcode == "V_FREXP_MANT_F16" ||
         opcode == "V_FREXP_EXP_I16_F16" ||
         opcode == "V_FRACT_F16" || opcode == "V_TRUNC_F16" ||
         opcode == "V_CEIL_F16" || opcode == "V_RNDNE_F16" ||
         opcode == "V_FLOOR_F16" ||
         opcode == "V_CVT_F32_I32" || opcode == "V_CVT_F32_U32" ||
         opcode == "V_CVT_U32_F32" || opcode == "V_CVT_I32_F32" ||
         opcode == "V_CVT_RPI_I32_F32" || opcode == "V_CVT_FLR_I32_F32" ||
         opcode == "V_CVT_I32_F64" || opcode == "V_CVT_U32_F64" ||
         opcode == "V_CVT_F16_F32" || opcode == "V_CVT_F32_F16" ||
         opcode == "V_CVT_F32_F64" || opcode == "V_CVT_F64_F32" ||
         opcode == "V_CVT_F64_I32" || opcode == "V_CVT_F64_U32" ||
         opcode == "V_EXP_F32" || opcode == "V_EXP_LEGACY_F32" ||
         opcode == "V_LOG_F32" || opcode == "V_LOG_LEGACY_F32" ||
         opcode == "V_RCP_F32" || opcode == "V_RCP_IFLAG_F32" ||
         opcode == "V_RSQ_F32" || opcode == "V_SQRT_F32" ||
         opcode == "V_SIN_F32" || opcode == "V_COS_F32" ||
         opcode == "V_FREXP_EXP_I32_F32" || opcode == "V_FREXP_MANT_F32" ||
         opcode == "V_FRACT_F32" || opcode == "V_TRUNC_F32" ||
         opcode == "V_CEIL_F32" || opcode == "V_RNDNE_F32" ||
         opcode == "V_FLOOR_F32" ||
         opcode == "V_RCP_F64" || opcode == "V_RSQ_F64" ||
         opcode == "V_SQRT_F64" || opcode == "V_FREXP_EXP_I32_F64" ||
         opcode == "V_FREXP_MANT_F64" || opcode == "V_FRACT_F64" ||
         opcode == "V_TRUNC_F64" || opcode == "V_CEIL_F64" ||
         opcode == "V_RNDNE_F64" || opcode == "V_FLOOR_F64";
}

bool IsVectorToScalarOpcode(std::string_view opcode) {
  return opcode == "V_READFIRSTLANE_B32" || opcode == "V_READLANE_B32";
}

bool IsVectorTernaryOpcode(std::string_view opcode) {
  return opcode == "V_ADD3_U32" || opcode == "V_LSHL_ADD_U32" ||
         opcode == "V_LSHL_ADD_U64" ||
         opcode == "V_FMA_F32" || opcode == "V_FMA_F64" ||
         opcode == "V_ADD_LSHL_U32" || opcode == "V_LSHL_OR_B32" ||
         opcode == "V_AND_OR_B32" || opcode == "V_OR3_B32" ||
         opcode == "V_XAD_U32" ||
         opcode == "V_LERP_U8" ||
         opcode == "V_PERM_B32" ||
         opcode == "V_BFE_U32" ||
         opcode == "V_BFE_I32" || opcode == "V_BFI_B32" ||
         opcode == "V_ALIGNBIT_B32" || opcode == "V_ALIGNBYTE_B32" ||
         opcode == "V_MIN3_I32" || opcode == "V_MIN3_U32" ||
         opcode == "V_MAX3_I32" || opcode == "V_MAX3_U32" ||
         opcode == "V_MED3_I32" || opcode == "V_MED3_U32" ||
         opcode == "V_SAD_U8" || opcode == "V_SAD_HI_U8" ||
         opcode == "V_SAD_U16" || opcode == "V_SAD_U32" ||
         opcode == "V_MAD_I32_I24" || opcode == "V_MAD_U32_U24" ||
         opcode == "V_MAD_U64_U32" || opcode == "V_MAD_I64_I32";
}

bool IsVectorCompareOpcode(std::string_view opcode) {
  return opcode == "V_CMP_F_F16" || opcode == "V_CMP_LT_F16" ||
         opcode == "V_CMP_EQ_F16" || opcode == "V_CMP_LE_F16" ||
         opcode == "V_CMP_GT_F16" || opcode == "V_CMP_LG_F16" ||
         opcode == "V_CMP_GE_F16" || opcode == "V_CMP_O_F16" ||
         opcode == "V_CMP_U_F16" || opcode == "V_CMP_NGE_F16" ||
         opcode == "V_CMP_NLG_F16" || opcode == "V_CMP_NGT_F16" ||
         opcode == "V_CMP_NLE_F16" || opcode == "V_CMP_NEQ_F16" ||
         opcode == "V_CMP_NLT_F16" || opcode == "V_CMP_TRU_F16" ||
         opcode == "V_CMP_CLASS_F16" ||
         opcode == "V_CMP_F_F32" || opcode == "V_CMP_LT_F32" ||
         opcode == "V_CMP_EQ_F32" || opcode == "V_CMP_LE_F32" ||
         opcode == "V_CMP_GT_F32" || opcode == "V_CMP_LG_F32" ||
         opcode == "V_CMP_GE_F32" || opcode == "V_CMP_O_F32" ||
         opcode == "V_CMP_U_F32" || opcode == "V_CMP_NGE_F32" ||
         opcode == "V_CMP_NLG_F32" || opcode == "V_CMP_NGT_F32" ||
         opcode == "V_CMP_NLE_F32" || opcode == "V_CMP_NEQ_F32" ||
         opcode == "V_CMP_NLT_F32" || opcode == "V_CMP_TRU_F32" ||
         opcode == "V_CMP_CLASS_F32" ||
         opcode == "V_CMP_F_F64" || opcode == "V_CMP_LT_F64" ||
         opcode == "V_CMP_EQ_F64" || opcode == "V_CMP_LE_F64" ||
         opcode == "V_CMP_GT_F64" || opcode == "V_CMP_LG_F64" ||
         opcode == "V_CMP_GE_F64" || opcode == "V_CMP_O_F64" ||
         opcode == "V_CMP_U_F64" || opcode == "V_CMP_NGE_F64" ||
         opcode == "V_CMP_NLG_F64" || opcode == "V_CMP_NGT_F64" ||
         opcode == "V_CMP_NLE_F64" || opcode == "V_CMP_NEQ_F64" ||
         opcode == "V_CMP_NLT_F64" || opcode == "V_CMP_TRU_F64" ||
         opcode == "V_CMP_CLASS_F64" ||
         opcode == "V_CMPX_F_F16" || opcode == "V_CMPX_LT_F16" ||
         opcode == "V_CMPX_EQ_F16" || opcode == "V_CMPX_LE_F16" ||
         opcode == "V_CMPX_GT_F16" || opcode == "V_CMPX_LG_F16" ||
         opcode == "V_CMPX_GE_F16" || opcode == "V_CMPX_O_F16" ||
         opcode == "V_CMPX_U_F16" || opcode == "V_CMPX_NGE_F16" ||
         opcode == "V_CMPX_NLG_F16" || opcode == "V_CMPX_NGT_F16" ||
         opcode == "V_CMPX_NLE_F16" || opcode == "V_CMPX_NEQ_F16" ||
         opcode == "V_CMPX_NLT_F16" || opcode == "V_CMPX_TRU_F16" ||
         opcode == "V_CMPX_CLASS_F16" ||
         opcode == "V_CMPX_F_F32" || opcode == "V_CMPX_LT_F32" ||
         opcode == "V_CMPX_EQ_F32" || opcode == "V_CMPX_LE_F32" ||
         opcode == "V_CMPX_GT_F32" || opcode == "V_CMPX_LG_F32" ||
         opcode == "V_CMPX_GE_F32" || opcode == "V_CMPX_O_F32" ||
         opcode == "V_CMPX_U_F32" || opcode == "V_CMPX_NGE_F32" ||
         opcode == "V_CMPX_NLG_F32" || opcode == "V_CMPX_NGT_F32" ||
         opcode == "V_CMPX_NLE_F32" || opcode == "V_CMPX_NEQ_F32" ||
         opcode == "V_CMPX_NLT_F32" || opcode == "V_CMPX_TRU_F32" ||
         opcode == "V_CMPX_CLASS_F32" ||
         opcode == "V_CMPX_F_F64" || opcode == "V_CMPX_LT_F64" ||
         opcode == "V_CMPX_EQ_F64" || opcode == "V_CMPX_LE_F64" ||
         opcode == "V_CMPX_GT_F64" || opcode == "V_CMPX_LG_F64" ||
         opcode == "V_CMPX_GE_F64" || opcode == "V_CMPX_O_F64" ||
         opcode == "V_CMPX_U_F64" || opcode == "V_CMPX_NGE_F64" ||
         opcode == "V_CMPX_NLG_F64" || opcode == "V_CMPX_NGT_F64" ||
         opcode == "V_CMPX_NLE_F64" || opcode == "V_CMPX_NEQ_F64" ||
         opcode == "V_CMPX_NLT_F64" || opcode == "V_CMPX_TRU_F64" ||
         opcode == "V_CMPX_CLASS_F64" ||
         opcode == "V_CMP_EQ_I32" || opcode == "V_CMP_NE_I32" ||
         opcode == "V_CMP_LT_I32" || opcode == "V_CMP_LE_I32" ||
         opcode == "V_CMP_GT_I32" || opcode == "V_CMP_GE_I32" ||
         opcode == "V_CMP_EQ_U32" || opcode == "V_CMP_NE_U32" ||
         opcode == "V_CMP_LT_U32" || opcode == "V_CMP_LE_U32" ||
         opcode == "V_CMP_GT_U32" || opcode == "V_CMP_GE_U32" ||
         opcode == "V_CMPX_F_I32" || opcode == "V_CMPX_LT_I32" ||
         opcode == "V_CMPX_EQ_I32" || opcode == "V_CMPX_LE_I32" ||
         opcode == "V_CMPX_GT_I32" || opcode == "V_CMPX_NE_I32" ||
         opcode == "V_CMPX_GE_I32" || opcode == "V_CMPX_T_I32" ||
         opcode == "V_CMPX_F_U32" || opcode == "V_CMPX_LT_U32" ||
         opcode == "V_CMPX_EQ_U32" || opcode == "V_CMPX_LE_U32" ||
         opcode == "V_CMPX_GT_U32" || opcode == "V_CMPX_NE_U32" ||
         opcode == "V_CMPX_GE_U32" || opcode == "V_CMPX_T_U32" ||
         opcode == "V_CMP_F_I64" || opcode == "V_CMP_LT_I64" ||
         opcode == "V_CMP_EQ_I64" || opcode == "V_CMP_LE_I64" ||
         opcode == "V_CMP_GT_I64" || opcode == "V_CMP_NE_I64" ||
         opcode == "V_CMP_GE_I64" || opcode == "V_CMP_T_I64" ||
         opcode == "V_CMP_F_U64" || opcode == "V_CMP_LT_U64" ||
         opcode == "V_CMP_EQ_U64" || opcode == "V_CMP_LE_U64" ||
         opcode == "V_CMP_GT_U64" || opcode == "V_CMP_NE_U64" ||
         opcode == "V_CMP_GE_U64" || opcode == "V_CMP_T_U64" ||
         opcode == "V_CMPX_F_I64" || opcode == "V_CMPX_LT_I64" ||
         opcode == "V_CMPX_EQ_I64" || opcode == "V_CMPX_LE_I64" ||
         opcode == "V_CMPX_GT_I64" || opcode == "V_CMPX_NE_I64" ||
         opcode == "V_CMPX_GE_I64" || opcode == "V_CMPX_T_I64" ||
         opcode == "V_CMPX_F_U64" || opcode == "V_CMPX_LT_U64" ||
         opcode == "V_CMPX_EQ_U64" || opcode == "V_CMPX_LE_U64" ||
         opcode == "V_CMPX_GT_U64" || opcode == "V_CMPX_NE_U64" ||
         opcode == "V_CMPX_GE_U64" || opcode == "V_CMPX_T_U64";
}

bool IsVectorCmpxOpcode(std::string_view opcode) {
  return opcode == "V_CMPX_F_F16" || opcode == "V_CMPX_LT_F16" ||
         opcode == "V_CMPX_EQ_F16" || opcode == "V_CMPX_LE_F16" ||
         opcode == "V_CMPX_GT_F16" || opcode == "V_CMPX_LG_F16" ||
         opcode == "V_CMPX_GE_F16" || opcode == "V_CMPX_O_F16" ||
         opcode == "V_CMPX_U_F16" || opcode == "V_CMPX_NGE_F16" ||
         opcode == "V_CMPX_NLG_F16" || opcode == "V_CMPX_NGT_F16" ||
         opcode == "V_CMPX_NLE_F16" || opcode == "V_CMPX_NEQ_F16" ||
         opcode == "V_CMPX_NLT_F16" || opcode == "V_CMPX_TRU_F16" ||
         opcode == "V_CMPX_CLASS_F16" ||
         opcode == "V_CMPX_F_F32" || opcode == "V_CMPX_LT_F32" ||
         opcode == "V_CMPX_EQ_F32" || opcode == "V_CMPX_LE_F32" ||
         opcode == "V_CMPX_GT_F32" || opcode == "V_CMPX_LG_F32" ||
         opcode == "V_CMPX_GE_F32" || opcode == "V_CMPX_O_F32" ||
         opcode == "V_CMPX_U_F32" || opcode == "V_CMPX_NGE_F32" ||
         opcode == "V_CMPX_NLG_F32" || opcode == "V_CMPX_NGT_F32" ||
         opcode == "V_CMPX_NLE_F32" || opcode == "V_CMPX_NEQ_F32" ||
         opcode == "V_CMPX_NLT_F32" || opcode == "V_CMPX_TRU_F32" ||
         opcode == "V_CMPX_CLASS_F32" ||
         opcode == "V_CMPX_F_F64" || opcode == "V_CMPX_LT_F64" ||
         opcode == "V_CMPX_EQ_F64" || opcode == "V_CMPX_LE_F64" ||
         opcode == "V_CMPX_GT_F64" || opcode == "V_CMPX_LG_F64" ||
         opcode == "V_CMPX_GE_F64" || opcode == "V_CMPX_O_F64" ||
         opcode == "V_CMPX_U_F64" || opcode == "V_CMPX_NGE_F64" ||
         opcode == "V_CMPX_NLG_F64" || opcode == "V_CMPX_NGT_F64" ||
         opcode == "V_CMPX_NLE_F64" || opcode == "V_CMPX_NEQ_F64" ||
         opcode == "V_CMPX_NLT_F64" || opcode == "V_CMPX_TRU_F64" ||
         opcode == "V_CMPX_CLASS_F64" ||
         opcode == "V_CMPX_F_I32" || opcode == "V_CMPX_LT_I32" ||
         opcode == "V_CMPX_EQ_I32" || opcode == "V_CMPX_LE_I32" ||
         opcode == "V_CMPX_GT_I32" || opcode == "V_CMPX_NE_I32" ||
         opcode == "V_CMPX_GE_I32" || opcode == "V_CMPX_T_I32" ||
         opcode == "V_CMPX_F_U32" || opcode == "V_CMPX_LT_U32" ||
         opcode == "V_CMPX_EQ_U32" || opcode == "V_CMPX_LE_U32" ||
         opcode == "V_CMPX_GT_U32" || opcode == "V_CMPX_NE_U32" ||
         opcode == "V_CMPX_GE_U32" || opcode == "V_CMPX_T_U32" ||
         opcode == "V_CMPX_F_I64" || opcode == "V_CMPX_LT_I64" ||
         opcode == "V_CMPX_EQ_I64" || opcode == "V_CMPX_LE_I64" ||
         opcode == "V_CMPX_GT_I64" || opcode == "V_CMPX_NE_I64" ||
         opcode == "V_CMPX_GE_I64" || opcode == "V_CMPX_T_I64" ||
         opcode == "V_CMPX_F_U64" || opcode == "V_CMPX_LT_U64" ||
         opcode == "V_CMPX_EQ_U64" || opcode == "V_CMPX_LE_U64" ||
         opcode == "V_CMPX_GT_U64" || opcode == "V_CMPX_NE_U64" ||
         opcode == "V_CMPX_GE_U64" || opcode == "V_CMPX_T_U64";
}

bool IsVectorCompare64Opcode(std::string_view opcode) {
  return opcode == "V_CMP_F_F64" || opcode == "V_CMP_LT_F64" ||
         opcode == "V_CMP_EQ_F64" || opcode == "V_CMP_LE_F64" ||
         opcode == "V_CMP_GT_F64" || opcode == "V_CMP_LG_F64" ||
         opcode == "V_CMP_GE_F64" || opcode == "V_CMP_O_F64" ||
         opcode == "V_CMP_U_F64" || opcode == "V_CMP_NGE_F64" ||
         opcode == "V_CMP_NLG_F64" || opcode == "V_CMP_NGT_F64" ||
         opcode == "V_CMP_NLE_F64" || opcode == "V_CMP_NEQ_F64" ||
         opcode == "V_CMP_NLT_F64" || opcode == "V_CMP_TRU_F64" ||
         opcode == "V_CMPX_F_F64" || opcode == "V_CMPX_LT_F64" ||
         opcode == "V_CMPX_EQ_F64" || opcode == "V_CMPX_LE_F64" ||
         opcode == "V_CMPX_GT_F64" || opcode == "V_CMPX_LG_F64" ||
         opcode == "V_CMPX_GE_F64" || opcode == "V_CMPX_O_F64" ||
         opcode == "V_CMPX_U_F64" || opcode == "V_CMPX_NGE_F64" ||
         opcode == "V_CMPX_NLG_F64" || opcode == "V_CMPX_NGT_F64" ||
         opcode == "V_CMPX_NLE_F64" || opcode == "V_CMPX_NEQ_F64" ||
         opcode == "V_CMPX_NLT_F64" || opcode == "V_CMPX_TRU_F64" ||
         opcode == "V_CMP_F_I64" || opcode == "V_CMP_LT_I64" ||
         opcode == "V_CMP_EQ_I64" || opcode == "V_CMP_LE_I64" ||
         opcode == "V_CMP_GT_I64" || opcode == "V_CMP_NE_I64" ||
         opcode == "V_CMP_GE_I64" || opcode == "V_CMP_T_I64" ||
         opcode == "V_CMP_F_U64" || opcode == "V_CMP_LT_U64" ||
         opcode == "V_CMP_EQ_U64" || opcode == "V_CMP_LE_U64" ||
         opcode == "V_CMP_GT_U64" || opcode == "V_CMP_NE_U64" ||
         opcode == "V_CMP_GE_U64" || opcode == "V_CMP_T_U64" ||
         opcode == "V_CMPX_F_I64" || opcode == "V_CMPX_LT_I64" ||
         opcode == "V_CMPX_EQ_I64" || opcode == "V_CMPX_LE_I64" ||
         opcode == "V_CMPX_GT_I64" || opcode == "V_CMPX_NE_I64" ||
         opcode == "V_CMPX_GE_I64" || opcode == "V_CMPX_T_I64" ||
         opcode == "V_CMPX_F_U64" || opcode == "V_CMPX_LT_U64" ||
         opcode == "V_CMPX_EQ_U64" || opcode == "V_CMPX_LE_U64" ||
         opcode == "V_CMPX_GT_U64" || opcode == "V_CMPX_NE_U64" ||
         opcode == "V_CMPX_GE_U64" || opcode == "V_CMPX_T_U64";
}

bool IsVectorCompareClassOpcode(std::string_view opcode) {
  return opcode == "V_CMP_CLASS_F16" || opcode == "V_CMP_CLASS_F32" ||
         opcode == "V_CMP_CLASS_F64" || opcode == "V_CMPX_CLASS_F16" ||
         opcode == "V_CMPX_CLASS_F32" || opcode == "V_CMPX_CLASS_F64";
}

bool IsVectorCompareClass64Opcode(std::string_view opcode) {
  return opcode == "V_CMP_CLASS_F64" || opcode == "V_CMPX_CLASS_F64";
}

bool IsVectorCmpxOpcode(CompiledOpcode opcode) {
  switch (opcode) {
    case CompiledOpcode::kVCmpxFF16:
    case CompiledOpcode::kVCmpxLtF16:
    case CompiledOpcode::kVCmpxEqF16:
    case CompiledOpcode::kVCmpxLeF16:
    case CompiledOpcode::kVCmpxGtF16:
    case CompiledOpcode::kVCmpxLgF16:
    case CompiledOpcode::kVCmpxGeF16:
    case CompiledOpcode::kVCmpxOF16:
    case CompiledOpcode::kVCmpxUF16:
    case CompiledOpcode::kVCmpxNgeF16:
    case CompiledOpcode::kVCmpxNlgF16:
    case CompiledOpcode::kVCmpxNgtF16:
    case CompiledOpcode::kVCmpxNleF16:
    case CompiledOpcode::kVCmpxNeqF16:
    case CompiledOpcode::kVCmpxNltF16:
    case CompiledOpcode::kVCmpxTruF16:
    case CompiledOpcode::kVCmpxClassF16:
    case CompiledOpcode::kVCmpxFF32:
    case CompiledOpcode::kVCmpxLtF32:
    case CompiledOpcode::kVCmpxEqF32:
    case CompiledOpcode::kVCmpxLeF32:
    case CompiledOpcode::kVCmpxGtF32:
    case CompiledOpcode::kVCmpxLgF32:
    case CompiledOpcode::kVCmpxGeF32:
    case CompiledOpcode::kVCmpxOF32:
    case CompiledOpcode::kVCmpxUF32:
    case CompiledOpcode::kVCmpxNgeF32:
    case CompiledOpcode::kVCmpxNlgF32:
    case CompiledOpcode::kVCmpxNgtF32:
    case CompiledOpcode::kVCmpxNleF32:
    case CompiledOpcode::kVCmpxNeqF32:
    case CompiledOpcode::kVCmpxNltF32:
    case CompiledOpcode::kVCmpxTruF32:
    case CompiledOpcode::kVCmpxClassF32:
    case CompiledOpcode::kVCmpxFF64:
    case CompiledOpcode::kVCmpxLtF64:
    case CompiledOpcode::kVCmpxEqF64:
    case CompiledOpcode::kVCmpxLeF64:
    case CompiledOpcode::kVCmpxGtF64:
    case CompiledOpcode::kVCmpxLgF64:
    case CompiledOpcode::kVCmpxGeF64:
    case CompiledOpcode::kVCmpxOF64:
    case CompiledOpcode::kVCmpxUF64:
    case CompiledOpcode::kVCmpxNgeF64:
    case CompiledOpcode::kVCmpxNlgF64:
    case CompiledOpcode::kVCmpxNgtF64:
    case CompiledOpcode::kVCmpxNleF64:
    case CompiledOpcode::kVCmpxNeqF64:
    case CompiledOpcode::kVCmpxNltF64:
    case CompiledOpcode::kVCmpxTruF64:
    case CompiledOpcode::kVCmpxClassF64:
    case CompiledOpcode::kVCmpxFI32:
    case CompiledOpcode::kVCmpxLtI32:
    case CompiledOpcode::kVCmpxEqI32:
    case CompiledOpcode::kVCmpxLeI32:
    case CompiledOpcode::kVCmpxGtI32:
    case CompiledOpcode::kVCmpxNeI32:
    case CompiledOpcode::kVCmpxGeI32:
    case CompiledOpcode::kVCmpxTI32:
    case CompiledOpcode::kVCmpxFU32:
    case CompiledOpcode::kVCmpxLtU32:
    case CompiledOpcode::kVCmpxEqU32:
    case CompiledOpcode::kVCmpxLeU32:
    case CompiledOpcode::kVCmpxGtU32:
    case CompiledOpcode::kVCmpxNeU32:
    case CompiledOpcode::kVCmpxGeU32:
    case CompiledOpcode::kVCmpxTU32:
    case CompiledOpcode::kVCmpxFI64:
    case CompiledOpcode::kVCmpxLtI64:
    case CompiledOpcode::kVCmpxEqI64:
    case CompiledOpcode::kVCmpxLeI64:
    case CompiledOpcode::kVCmpxGtI64:
    case CompiledOpcode::kVCmpxNeI64:
    case CompiledOpcode::kVCmpxGeI64:
    case CompiledOpcode::kVCmpxTI64:
    case CompiledOpcode::kVCmpxFU64:
    case CompiledOpcode::kVCmpxLtU64:
    case CompiledOpcode::kVCmpxEqU64:
    case CompiledOpcode::kVCmpxLeU64:
    case CompiledOpcode::kVCmpxGtU64:
    case CompiledOpcode::kVCmpxNeU64:
    case CompiledOpcode::kVCmpxGeU64:
    case CompiledOpcode::kVCmpxTU64:
      return true;
    default:
      return false;
  }
}

bool IsVectorCompareClassOpcode(CompiledOpcode opcode) {
  switch (opcode) {
    case CompiledOpcode::kVCmpClassF16:
    case CompiledOpcode::kVCmpClassF32:
    case CompiledOpcode::kVCmpClassF64:
    case CompiledOpcode::kVCmpxClassF16:
    case CompiledOpcode::kVCmpxClassF32:
    case CompiledOpcode::kVCmpxClassF64:
      return true;
    default:
      return false;
  }
}

bool IsVectorCompareClass64Opcode(CompiledOpcode opcode) {
  return opcode == CompiledOpcode::kVCmpClassF64 ||
         opcode == CompiledOpcode::kVCmpxClassF64;
}

bool IsFlatVectorMemoryOpcode(std::string_view opcode) {
  return opcode == "FLAT_LOAD_UBYTE" || opcode == "FLAT_LOAD_SBYTE" ||
         opcode == "FLAT_LOAD_USHORT" || opcode == "FLAT_LOAD_SSHORT" ||
         opcode == "FLAT_LOAD_DWORD" || opcode == "FLAT_LOAD_DWORDX2" ||
         opcode == "FLAT_LOAD_DWORDX3" || opcode == "FLAT_LOAD_DWORDX4" ||
         opcode == "FLAT_STORE_BYTE" || opcode == "FLAT_STORE_SHORT" ||
         opcode == "FLAT_STORE_DWORD" || opcode == "FLAT_STORE_DWORDX2" ||
         opcode == "FLAT_STORE_DWORDX3" || opcode == "FLAT_STORE_DWORDX4";
}

bool IsGlobalVectorMemoryOpcode(std::string_view opcode) {
  return opcode == "GLOBAL_LOAD_UBYTE" || opcode == "GLOBAL_LOAD_SBYTE" ||
         opcode == "GLOBAL_LOAD_USHORT" || opcode == "GLOBAL_LOAD_SSHORT" ||
         opcode == "GLOBAL_LOAD_DWORD" || opcode == "GLOBAL_LOAD_DWORDX2" ||
         opcode == "GLOBAL_LOAD_DWORDX3" || opcode == "GLOBAL_LOAD_DWORDX4" ||
         opcode == "GLOBAL_STORE_BYTE" || opcode == "GLOBAL_STORE_SHORT" ||
         opcode == "GLOBAL_STORE_DWORD" || opcode == "GLOBAL_STORE_DWORDX2" ||
         opcode == "GLOBAL_STORE_DWORDX3" || opcode == "GLOBAL_STORE_DWORDX4";
}

bool IsVectorMemoryOpcode(std::string_view opcode) {
  return IsFlatVectorMemoryOpcode(opcode) || IsGlobalVectorMemoryOpcode(opcode);
}

bool IsDsOpcode(std::string_view opcode) {
  return opcode == "DS_NOP" ||
         opcode == "DS_WRITE_B32" || opcode == "DS_READ_B32" ||
         opcode == "DS_ADD_U32" || opcode == "DS_SUB_U32" ||
         opcode == "DS_RSUB_U32" || opcode == "DS_INC_U32" ||
         opcode == "DS_DEC_U32" || opcode == "DS_MIN_I32" ||
         opcode == "DS_MAX_I32" || opcode == "DS_MIN_U32" ||
         opcode == "DS_MAX_U32" || opcode == "DS_AND_B32" ||
         opcode == "DS_OR_B32" || opcode == "DS_XOR_B32" ||
         opcode == "DS_MSKOR_B32" || opcode == "DS_CMPST_B32" ||
         opcode == "DS_CMPST_F32" ||
         opcode == "DS_ADD_F32" || opcode == "DS_MIN_F32" ||
         opcode == "DS_MAX_F32" || opcode == "DS_WRITE_B8" ||
         opcode == "DS_WRITE_B16" || opcode == "DS_WRITE2_B32" ||
         opcode == "DS_WRITE2ST64_B32" || opcode == "DS_READ2_B32" ||
         opcode == "DS_READ2ST64_B32" || opcode == "DS_READ_I8" ||
         opcode == "DS_READ_U8" || opcode == "DS_READ_I16" ||
         opcode == "DS_READ_U16" ||
         opcode == "DS_ADD_RTN_U32" || opcode == "DS_SUB_RTN_U32" ||
         opcode == "DS_RSUB_RTN_U32" || opcode == "DS_INC_RTN_U32" ||
         opcode == "DS_DEC_RTN_U32" || opcode == "DS_MIN_RTN_I32" ||
         opcode == "DS_MAX_RTN_I32" || opcode == "DS_MIN_RTN_U32" ||
         opcode == "DS_MAX_RTN_U32" || opcode == "DS_AND_RTN_B32" ||
         opcode == "DS_OR_RTN_B32" || opcode == "DS_XOR_RTN_B32" ||
         opcode == "DS_MSKOR_RTN_B32" || opcode == "DS_WRXCHG_RTN_B32" ||
         opcode == "DS_CMPST_RTN_B32" || opcode == "DS_CMPST_RTN_F32" ||
         opcode == "DS_WRAP_RTN_B32" ||
         opcode == "DS_ADD_RTN_F32" || opcode == "DS_MIN_RTN_F32" ||
         opcode == "DS_MAX_RTN_F32";
}

bool IsDsPairWriteOpcode(std::string_view opcode) {
  return opcode == "DS_WRITE2_B32" || opcode == "DS_WRITE2ST64_B32";
}

bool IsDsPairReadOpcode(std::string_view opcode) {
  return opcode == "DS_READ2_B32" || opcode == "DS_READ2ST64_B32";
}

bool IsDsNarrowReadOpcode(std::string_view opcode) {
  return opcode == "DS_READ_I8" || opcode == "DS_READ_U8" ||
         opcode == "DS_READ_I16" || opcode == "DS_READ_U16";
}

bool IsDsDualDataOpcode(std::string_view opcode) {
  return opcode == "DS_MSKOR_B32" || opcode == "DS_CMPST_B32" ||
         opcode == "DS_CMPST_F32";
}

bool IsDsDualDataReturnOpcode(std::string_view opcode) {
  return opcode == "DS_MSKOR_RTN_B32" || opcode == "DS_CMPST_RTN_B32" ||
         opcode == "DS_CMPST_RTN_F32" || opcode == "DS_WRAP_RTN_B32";
}

bool IsDsSignedReadOpcode(std::string_view opcode) {
  return opcode == "DS_READ_I8" || opcode == "DS_READ_I16";
}

bool IsDsReturnOpcode(std::string_view opcode) {
  return opcode == "DS_ADD_RTN_U32" || opcode == "DS_SUB_RTN_U32" ||
         opcode == "DS_RSUB_RTN_U32" || opcode == "DS_INC_RTN_U32" ||
         opcode == "DS_DEC_RTN_U32" || opcode == "DS_MIN_RTN_I32" ||
         opcode == "DS_MAX_RTN_I32" || opcode == "DS_MIN_RTN_U32" ||
         opcode == "DS_MAX_RTN_U32" || opcode == "DS_AND_RTN_B32" ||
         opcode == "DS_OR_RTN_B32" || opcode == "DS_XOR_RTN_B32" ||
         opcode == "DS_WRXCHG_RTN_B32" ||
         opcode == "DS_ADD_RTN_F32" || opcode == "DS_MIN_RTN_F32" ||
         opcode == "DS_MAX_RTN_F32";
}

std::size_t GetDsAccessSize(std::string_view opcode) {
  if (opcode == "DS_WRITE_B8" || opcode == "DS_READ_I8" ||
      opcode == "DS_READ_U8") {
    return 1;
  }
  if (opcode == "DS_WRITE_B16" || opcode == "DS_READ_I16" ||
      opcode == "DS_READ_U16") {
    return 2;
  }
  return sizeof(std::uint32_t);
}

std::size_t GetDsAccessSize(CompiledOpcode opcode) {
  switch (opcode) {
    case CompiledOpcode::kDsWriteB8:
    case CompiledOpcode::kDsReadI8:
    case CompiledOpcode::kDsReadU8:
      return 1;
    case CompiledOpcode::kDsWriteB16:
    case CompiledOpcode::kDsReadI16:
    case CompiledOpcode::kDsReadU16:
      return 2;
    default:
      return sizeof(std::uint32_t);
  }
}

bool IsDsPairWriteOpcode(CompiledOpcode opcode) {
  return opcode == CompiledOpcode::kDsWrite2B32 ||
         opcode == CompiledOpcode::kDsWrite2St64B32;
}

bool IsDsPairReadOpcode(CompiledOpcode opcode) {
  return opcode == CompiledOpcode::kDsRead2B32 ||
         opcode == CompiledOpcode::kDsRead2St64B32;
}

bool IsDsNarrowReadOpcode(CompiledOpcode opcode) {
  return opcode == CompiledOpcode::kDsReadI8 ||
         opcode == CompiledOpcode::kDsReadU8 ||
         opcode == CompiledOpcode::kDsReadI16 ||
         opcode == CompiledOpcode::kDsReadU16;
}

bool IsDsDualDataOpcode(CompiledOpcode opcode) {
  return opcode == CompiledOpcode::kDsMskorB32 ||
         opcode == CompiledOpcode::kDsCmpstB32 ||
         opcode == CompiledOpcode::kDsCmpstF32;
}

bool IsDsDualDataReturnOpcode(CompiledOpcode opcode) {
  return opcode == CompiledOpcode::kDsMskorRtnB32 ||
         opcode == CompiledOpcode::kDsCmpstRtnB32 ||
         opcode == CompiledOpcode::kDsCmpstRtnF32 ||
         opcode == CompiledOpcode::kDsWrapRtnB32;
}

bool IsDsSignedReadOpcode(CompiledOpcode opcode) {
  return opcode == CompiledOpcode::kDsReadI8 ||
         opcode == CompiledOpcode::kDsReadI16;
}

std::uint32_t GetDsPairOffsetScale(std::string_view opcode) {
  return opcode == "DS_WRITE2ST64_B32" || opcode == "DS_READ2ST64_B32" ? 256u
                                                                        : 4u;
}

std::uint32_t GetDsPairOffsetScale(CompiledOpcode opcode) {
  return opcode == CompiledOpcode::kDsWrite2St64B32 ||
                 opcode == CompiledOpcode::kDsRead2St64B32
             ? 256u
             : 4u;
}

bool IsDsReturnOpcode(CompiledOpcode opcode) {
  switch (opcode) {
    case CompiledOpcode::kDsAddRtnU32:
    case CompiledOpcode::kDsSubRtnU32:
    case CompiledOpcode::kDsRsubRtnU32:
    case CompiledOpcode::kDsIncRtnU32:
    case CompiledOpcode::kDsDecRtnU32:
    case CompiledOpcode::kDsMinRtnI32:
    case CompiledOpcode::kDsMaxRtnI32:
    case CompiledOpcode::kDsMinRtnU32:
    case CompiledOpcode::kDsMaxRtnU32:
    case CompiledOpcode::kDsAndRtnB32:
    case CompiledOpcode::kDsOrRtnB32:
    case CompiledOpcode::kDsXorRtnB32:
    case CompiledOpcode::kDsWrxchgRtnB32:
    case CompiledOpcode::kDsAddRtnF32:
    case CompiledOpcode::kDsMinRtnF32:
    case CompiledOpcode::kDsMaxRtnF32:
      return true;
    default:
      return false;
  }
}

std::string_view GetDsUpdateOpcode(std::string_view opcode) {
  if (opcode == "DS_ADD_RTN_U32") {
    return "DS_ADD_U32";
  }
  if (opcode == "DS_SUB_RTN_U32") {
    return "DS_SUB_U32";
  }
  if (opcode == "DS_RSUB_RTN_U32") {
    return "DS_RSUB_U32";
  }
  if (opcode == "DS_INC_RTN_U32") {
    return "DS_INC_U32";
  }
  if (opcode == "DS_DEC_RTN_U32") {
    return "DS_DEC_U32";
  }
  if (opcode == "DS_MIN_RTN_I32") {
    return "DS_MIN_I32";
  }
  if (opcode == "DS_MAX_RTN_I32") {
    return "DS_MAX_I32";
  }
  if (opcode == "DS_MIN_RTN_U32") {
    return "DS_MIN_U32";
  }
  if (opcode == "DS_MAX_RTN_U32") {
    return "DS_MAX_U32";
  }
  if (opcode == "DS_AND_RTN_B32") {
    return "DS_AND_B32";
  }
  if (opcode == "DS_OR_RTN_B32") {
    return "DS_OR_B32";
  }
  if (opcode == "DS_XOR_RTN_B32") {
    return "DS_XOR_B32";
  }
  if (opcode == "DS_WRXCHG_RTN_B32") {
    return "DS_WRITE_B32";
  }
  if (opcode == "DS_ADD_RTN_F32") {
    return "DS_ADD_F32";
  }
  if (opcode == "DS_MIN_RTN_F32") {
    return "DS_MIN_F32";
  }
  if (opcode == "DS_MAX_RTN_F32") {
    return "DS_MAX_F32";
  }
  return opcode;
}

CompiledOpcode GetDsUpdateOpcode(CompiledOpcode opcode) {
  switch (opcode) {
    case CompiledOpcode::kDsAddRtnU32:
      return CompiledOpcode::kDsAddU32;
    case CompiledOpcode::kDsSubRtnU32:
      return CompiledOpcode::kDsSubU32;
    case CompiledOpcode::kDsRsubRtnU32:
      return CompiledOpcode::kDsRsubU32;
    case CompiledOpcode::kDsIncRtnU32:
      return CompiledOpcode::kDsIncU32;
    case CompiledOpcode::kDsDecRtnU32:
      return CompiledOpcode::kDsDecU32;
    case CompiledOpcode::kDsMinRtnI32:
      return CompiledOpcode::kDsMinI32;
    case CompiledOpcode::kDsMaxRtnI32:
      return CompiledOpcode::kDsMaxI32;
    case CompiledOpcode::kDsMinRtnU32:
      return CompiledOpcode::kDsMinU32;
    case CompiledOpcode::kDsMaxRtnU32:
      return CompiledOpcode::kDsMaxU32;
    case CompiledOpcode::kDsAndRtnB32:
      return CompiledOpcode::kDsAndB32;
    case CompiledOpcode::kDsOrRtnB32:
      return CompiledOpcode::kDsOrB32;
    case CompiledOpcode::kDsXorRtnB32:
      return CompiledOpcode::kDsXorB32;
    case CompiledOpcode::kDsWrxchgRtnB32:
      return CompiledOpcode::kDsWriteB32;
    case CompiledOpcode::kDsAddRtnF32:
      return CompiledOpcode::kDsAddF32;
    case CompiledOpcode::kDsMinRtnF32:
      return CompiledOpcode::kDsMinF32;
    case CompiledOpcode::kDsMaxRtnF32:
      return CompiledOpcode::kDsMaxF32;
    default:
      return opcode;
  }
}

bool ComputeDsAddress(std::uint32_t base_address,
                      std::uint32_t offset,
                      std::uint32_t scale,
                      std::uint64_t* lds_address,
                      std::string* error_message) {
  if (lds_address == nullptr) {
    if (error_message != nullptr) {
      *error_message = "ds address output must not be null";
    }
    return false;
  }

  const std::uint64_t scaled_offset =
      static_cast<std::uint64_t>(offset) * static_cast<std::uint64_t>(scale);
  if (scaled_offset >
      std::numeric_limits<std::uint64_t>::max() -
          static_cast<std::uint64_t>(base_address)) {
    if (error_message != nullptr) {
      *error_message = "lds address overflow";
    }
    return false;
  }
  *lds_address = static_cast<std::uint64_t>(base_address) + scaled_offset;
  return true;
}

bool IsLdsAccessInBounds(std::span<std::byte> lds_storage,
                         std::uint64_t lds_address,
                         std::size_t access_size,
                         std::string* error_message) {
  if (lds_address > lds_storage.size() ||
      access_size > lds_storage.size() - static_cast<std::size_t>(lds_address)) {
    if (error_message != nullptr) {
      *error_message = "lds address out of range";
    }
    return false;
  }
  return true;
}

bool ReadLdsValue(std::span<std::byte> lds_storage,
                  std::uint64_t lds_address,
                  std::size_t access_size,
                  bool sign_extend,
                  std::uint32_t* value,
                  std::string* error_message) {
  if (value == nullptr) {
    if (error_message != nullptr) {
      *error_message = "lds read output must not be null";
    }
    return false;
  }
  if (!IsLdsAccessInBounds(lds_storage, lds_address, access_size, error_message)) {
    return false;
  }

  if (access_size == 1) {
    std::uint8_t raw_value = 0;
    std::memcpy(&raw_value, lds_storage.data() + lds_address, sizeof(raw_value));
    *value = sign_extend
                 ? static_cast<std::uint32_t>(
                       static_cast<std::int32_t>(static_cast<std::int8_t>(raw_value)))
                 : static_cast<std::uint32_t>(raw_value);
    return true;
  }
  if (access_size == 2) {
    std::uint16_t raw_value = 0;
    std::memcpy(&raw_value, lds_storage.data() + lds_address, sizeof(raw_value));
    *value =
        sign_extend
            ? static_cast<std::uint32_t>(static_cast<std::int32_t>(
                  static_cast<std::int16_t>(raw_value)))
            : static_cast<std::uint32_t>(raw_value);
    return true;
  }
  if (access_size == sizeof(std::uint32_t)) {
    std::memcpy(value, lds_storage.data() + lds_address, sizeof(*value));
    return true;
  }

  if (error_message != nullptr) {
    *error_message = "unsupported ds access size";
  }
  return false;
}

bool WriteLdsValue(std::span<std::byte> lds_storage,
                   std::uint64_t lds_address,
                   std::size_t access_size,
                   std::uint32_t value,
                   std::string* error_message) {
  if (!IsLdsAccessInBounds(lds_storage, lds_address, access_size, error_message)) {
    return false;
  }

  if (access_size == 1) {
    const std::uint8_t truncated = static_cast<std::uint8_t>(value);
    std::memcpy(lds_storage.data() + lds_address, &truncated, sizeof(truncated));
    return true;
  }
  if (access_size == 2) {
    const std::uint16_t truncated = static_cast<std::uint16_t>(value);
    std::memcpy(lds_storage.data() + lds_address, &truncated, sizeof(truncated));
    return true;
  }
  if (access_size == sizeof(std::uint32_t)) {
    std::memcpy(lds_storage.data() + lds_address, &value, sizeof(value));
    return true;
  }

  if (error_message != nullptr) {
    *error_message = "unsupported ds access size";
  }
  return false;
}

std::uint32_t EvaluateDsUpdate(std::string_view opcode,
                               std::uint32_t old_value,
                               std::uint32_t data_value,
                               std::string* error_message) {
  if (opcode == "DS_WRITE_B32") {
    return data_value;
  }
  if (opcode == "DS_ADD_U32") {
    return old_value + data_value;
  }
  if (opcode == "DS_SUB_U32") {
    return old_value - data_value;
  }
  if (opcode == "DS_RSUB_U32") {
    return data_value - old_value;
  }
  if (opcode == "DS_INC_U32") {
    return old_value >= data_value ? 0u : old_value + 1u;
  }
  if (opcode == "DS_DEC_U32") {
    return old_value == 0 ? data_value : old_value - 1u;
  }
  if (opcode == "DS_MIN_I32") {
    return BitCast<std::uint32_t>(
        std::min(BitCast<std::int32_t>(old_value),
                 BitCast<std::int32_t>(data_value)));
  }
  if (opcode == "DS_MAX_I32") {
    return BitCast<std::uint32_t>(
        std::max(BitCast<std::int32_t>(old_value),
                 BitCast<std::int32_t>(data_value)));
  }
  if (opcode == "DS_MIN_U32") {
    return std::min(old_value, data_value);
  }
  if (opcode == "DS_MAX_U32") {
    return std::max(old_value, data_value);
  }
  if (opcode == "DS_AND_B32") {
    return old_value & data_value;
  }
  if (opcode == "DS_OR_B32") {
    return old_value | data_value;
  }
  if (opcode == "DS_XOR_B32") {
    return old_value ^ data_value;
  }
  if (opcode == "DS_ADD_F32") {
    return BitCast<std::uint32_t>(BitCast<float>(old_value) +
                                  BitCast<float>(data_value));
  }
  if (opcode == "DS_MIN_F32") {
    return BitCast<std::uint32_t>(
        std::fmin(BitCast<float>(old_value), BitCast<float>(data_value)));
  }
  if (opcode == "DS_MAX_F32") {
    return BitCast<std::uint32_t>(
        std::fmax(BitCast<float>(old_value), BitCast<float>(data_value)));
  }

  if (error_message != nullptr) {
    *error_message = "unsupported ds opcode";
  }
  return 0;
}

bool DsCmpstF32Equal(std::uint32_t lhs_bits, std::uint32_t rhs_bits) {
  return BitCast<float>(lhs_bits) == BitCast<float>(rhs_bits);
}

std::uint32_t EvaluateDsDualDataUpdate(std::string_view opcode,
                                       std::uint32_t old_value,
                                       std::uint32_t data0_value,
                                       std::uint32_t data1_value,
                                       std::string* error_message) {
  if (opcode == "DS_MSKOR_B32" || opcode == "DS_MSKOR_RTN_B32") {
    return (old_value & ~data0_value) | data1_value;
  }
  if (opcode == "DS_CMPST_B32" || opcode == "DS_CMPST_RTN_B32") {
    return old_value == data0_value ? data1_value : old_value;
  }
  if (opcode == "DS_CMPST_F32" || opcode == "DS_CMPST_RTN_F32") {
    return DsCmpstF32Equal(old_value, data0_value) ? data1_value : old_value;
  }
  if (opcode == "DS_WRAP_RTN_B32") {
    return (old_value == 0u || old_value >= data1_value) ? data0_value
                                                          : old_value - 1u;
  }

  if (error_message != nullptr) {
    *error_message = "unsupported dual-data ds opcode";
  }
  return 0;
}

std::uint32_t EvaluateDsUpdate(CompiledOpcode opcode,
                               std::uint32_t old_value,
                               std::uint32_t data_value,
                               std::string* error_message) {
  switch (opcode) {
    case CompiledOpcode::kDsWriteB32:
      return data_value;
    case CompiledOpcode::kDsAddU32:
      return old_value + data_value;
    case CompiledOpcode::kDsSubU32:
      return old_value - data_value;
    case CompiledOpcode::kDsRsubU32:
      return data_value - old_value;
    case CompiledOpcode::kDsIncU32:
      return old_value >= data_value ? 0u : old_value + 1u;
    case CompiledOpcode::kDsDecU32:
      return old_value == 0 ? data_value : old_value - 1u;
    case CompiledOpcode::kDsMinI32:
      return BitCast<std::uint32_t>(
          std::min(BitCast<std::int32_t>(old_value),
                   BitCast<std::int32_t>(data_value)));
    case CompiledOpcode::kDsMaxI32:
      return BitCast<std::uint32_t>(
          std::max(BitCast<std::int32_t>(old_value),
                   BitCast<std::int32_t>(data_value)));
    case CompiledOpcode::kDsMinU32:
      return std::min(old_value, data_value);
    case CompiledOpcode::kDsMaxU32:
      return std::max(old_value, data_value);
    case CompiledOpcode::kDsAndB32:
      return old_value & data_value;
    case CompiledOpcode::kDsOrB32:
      return old_value | data_value;
    case CompiledOpcode::kDsXorB32:
      return old_value ^ data_value;
    case CompiledOpcode::kDsAddF32:
      return BitCast<std::uint32_t>(BitCast<float>(old_value) +
                                    BitCast<float>(data_value));
    case CompiledOpcode::kDsMinF32:
      return BitCast<std::uint32_t>(
          std::fmin(BitCast<float>(old_value), BitCast<float>(data_value)));
    case CompiledOpcode::kDsMaxF32:
      return BitCast<std::uint32_t>(
          std::fmax(BitCast<float>(old_value), BitCast<float>(data_value)));
    default:
      if (error_message != nullptr) {
        *error_message = "unsupported compiled ds opcode";
      }
      return 0;
  }
}

std::uint32_t EvaluateDsDualDataUpdate(CompiledOpcode opcode,
                                       std::uint32_t old_value,
                                       std::uint32_t data0_value,
                                       std::uint32_t data1_value,
                                       std::string* error_message) {
  switch (opcode) {
    case CompiledOpcode::kDsMskorB32:
    case CompiledOpcode::kDsMskorRtnB32:
      return (old_value & ~data0_value) | data1_value;
    case CompiledOpcode::kDsCmpstB32:
    case CompiledOpcode::kDsCmpstRtnB32:
      return old_value == data0_value ? data1_value : old_value;
    case CompiledOpcode::kDsCmpstF32:
    case CompiledOpcode::kDsCmpstRtnF32:
      return DsCmpstF32Equal(old_value, data0_value) ? data1_value : old_value;
    case CompiledOpcode::kDsWrapRtnB32:
      return (old_value == 0u || old_value >= data1_value) ? data0_value
                                                            : old_value - 1u;
    default:
      if (error_message != nullptr) {
        *error_message = "unsupported compiled dual-data ds opcode";
      }
      return 0;
  }
}

bool IsGlobalAtomicOpcode(std::string_view opcode) {
  return HasPrefix(opcode, "GLOBAL_ATOMIC_");
}

bool IsVectorMemoryStoreOpcode(std::string_view opcode) {
  return opcode == "FLAT_STORE_BYTE" || opcode == "FLAT_STORE_SHORT" ||
         opcode == "FLAT_STORE_DWORD" || opcode == "FLAT_STORE_DWORDX2" ||
         opcode == "FLAT_STORE_DWORDX3" || opcode == "FLAT_STORE_DWORDX4" ||
         opcode == "GLOBAL_STORE_BYTE" || opcode == "GLOBAL_STORE_SHORT" ||
         opcode == "GLOBAL_STORE_DWORD" || opcode == "GLOBAL_STORE_DWORDX2" ||
         opcode == "GLOBAL_STORE_DWORDX3" || opcode == "GLOBAL_STORE_DWORDX4";
}

bool IsVectorMemoryLoadOpcode(std::string_view opcode) {
  return opcode == "FLAT_LOAD_UBYTE" || opcode == "FLAT_LOAD_SBYTE" ||
         opcode == "FLAT_LOAD_USHORT" || opcode == "FLAT_LOAD_SSHORT" ||
         opcode == "FLAT_LOAD_DWORD" || opcode == "FLAT_LOAD_DWORDX2" ||
         opcode == "FLAT_LOAD_DWORDX3" || opcode == "FLAT_LOAD_DWORDX4" ||
         opcode == "GLOBAL_LOAD_UBYTE" || opcode == "GLOBAL_LOAD_SBYTE" ||
         opcode == "GLOBAL_LOAD_USHORT" || opcode == "GLOBAL_LOAD_SSHORT" ||
         opcode == "GLOBAL_LOAD_DWORD" || opcode == "GLOBAL_LOAD_DWORDX2" ||
         opcode == "GLOBAL_LOAD_DWORDX3" || opcode == "GLOBAL_LOAD_DWORDX4";
}

bool IsSignedVectorMemoryLoadOpcode(std::string_view opcode) {
  return opcode == "FLAT_LOAD_SBYTE" || opcode == "FLAT_LOAD_SSHORT" ||
         opcode == "GLOBAL_LOAD_SBYTE" || opcode == "GLOBAL_LOAD_SSHORT";
}

std::uint32_t ReverseBits32(std::uint32_t value) {
  value = ((value & 0x55555555u) << 1) | ((value >> 1) & 0x55555555u);
  value = ((value & 0x33333333u) << 2) | ((value >> 2) & 0x33333333u);
  value = ((value & 0x0f0f0f0fu) << 4) | ((value >> 4) & 0x0f0f0f0fu);
  value = ((value & 0x00ff00ffu) << 8) | ((value >> 8) & 0x00ff00ffu);
  return (value << 16) | (value >> 16);
}

std::uint32_t FindFirstBitHighUnsigned(std::uint32_t value) {
  if (value == 0u) {
    return 0xffffffffu;
  }
#if defined(__GNUC__) || defined(__clang__)
  return static_cast<std::uint32_t>(__builtin_clz(value));
#else
  std::uint32_t index = 0;
  while (((value >> (31u - index)) & 1u) == 0u) {
    ++index;
  }
  return index;
#endif
}

std::uint32_t FindFirstBitLow(std::uint32_t value) {
  if (value == 0u) {
    return 0xffffffffu;
  }
#if defined(__GNUC__) || defined(__clang__)
  return static_cast<std::uint32_t>(__builtin_ctz(value));
#else
  std::uint32_t index = 0;
  while (((value >> index) & 1u) == 0u) {
    ++index;
  }
  return index;
#endif
}

std::uint32_t FindFirstBitHighSigned(std::uint32_t value) {
  const bool sign = (value >> 31) != 0;
  const std::uint32_t toggled = sign ? ~value : value;
  if (toggled == 0u) {
    return 0xffffffffu;
  }
  return FindFirstBitHighUnsigned(toggled);
}

std::uint32_t FindFirstBitHighUnsigned64(std::uint64_t value) {
  if (value == 0u) {
    return 0xffffffffu;
  }
#if defined(__GNUC__) || defined(__clang__)
  return static_cast<std::uint32_t>(__builtin_clzll(value));
#else
  std::uint32_t index = 0;
  while (((value >> (63u - index)) & 1u) == 0u) {
    ++index;
  }
  return index;
#endif
}

std::uint32_t FindFirstBitLow64(std::uint64_t value) {
  if (value == 0u) {
    return 0xffffffffu;
  }
#if defined(__GNUC__) || defined(__clang__)
  return static_cast<std::uint32_t>(__builtin_ctzll(value));
#else
  std::uint32_t index = 0;
  while (((value >> index) & 1u) == 0u) {
    ++index;
  }
  return index;
#endif
}

std::uint32_t FindFirstBitHighSigned64(std::uint64_t value) {
  const bool sign = (value >> 63) != 0;
  const std::uint64_t toggled = sign ? ~value : value;
  if (toggled == 0u) {
    return 0xffffffffu;
  }
  return FindFirstBitHighUnsigned64(toggled);
}

std::uint64_t BitReplicate32To64(std::uint32_t value) {
  std::uint64_t result = 0;
  for (std::uint32_t bit = 0; bit < 32; ++bit) {
    if (((value >> bit) & 1u) != 0u) {
      result |= std::uint64_t{0x3} << (bit * 2u);
    }
  }
  return result;
}

std::uint64_t ReduceQuadMask(std::uint64_t value, unsigned bit_width) {
  std::uint64_t result = 0;
  for (unsigned quad_index = 0; quad_index < bit_width / 4; ++quad_index,
                value >>= 4) {
    if ((value & 0xfu) != 0u) {
      result |= std::uint64_t{1} << quad_index;
    }
  }
  return result;
}

std::uint32_t Min3Signed32(std::uint32_t a,
                           std::uint32_t b,
                           std::uint32_t c) {
  const std::int32_t sa = BitCast<std::int32_t>(a);
  const std::int32_t sb = BitCast<std::int32_t>(b);
  const std::int32_t sc = BitCast<std::int32_t>(c);
  return BitCast<std::uint32_t>(std::min({sa, sb, sc}));
}

std::uint32_t Max3Signed32(std::uint32_t a,
                           std::uint32_t b,
                           std::uint32_t c) {
  const std::int32_t sa = BitCast<std::int32_t>(a);
  const std::int32_t sb = BitCast<std::int32_t>(b);
  const std::int32_t sc = BitCast<std::int32_t>(c);
  return BitCast<std::uint32_t>(std::max({sa, sb, sc}));
}

std::uint32_t Med3Signed32(std::uint32_t a,
                           std::uint32_t b,
                           std::uint32_t c) {
  const std::int32_t sa = BitCast<std::int32_t>(a);
  const std::int32_t sb = BitCast<std::int32_t>(b);
  const std::int32_t sc = BitCast<std::int32_t>(c);
  return BitCast<std::uint32_t>(
      std::max(std::min(sa, sb), std::min(std::max(sa, sb), sc)));
}

std::uint32_t Min3Unsigned32(std::uint32_t a,
                             std::uint32_t b,
                             std::uint32_t c) {
  return std::min({a, b, c});
}

std::uint32_t Max3Unsigned32(std::uint32_t a,
                             std::uint32_t b,
                             std::uint32_t c) {
  return std::max({a, b, c});
}

std::uint32_t Med3Unsigned32(std::uint32_t a,
                             std::uint32_t b,
                             std::uint32_t c) {
  return std::max(std::min(a, b), std::min(std::max(a, b), c));
}

std::uint32_t SumAbsoluteDifferencesU8(std::uint32_t a, std::uint32_t b) {
  std::uint32_t sum = 0;
  for (unsigned byte_index = 0; byte_index < 4; ++byte_index) {
    const std::uint32_t lhs = (a >> (byte_index * 8u)) & 0xffu;
    const std::uint32_t rhs = (b >> (byte_index * 8u)) & 0xffu;
    sum += lhs >= rhs ? (lhs - rhs) : (rhs - lhs);
  }
  return sum;
}

std::uint32_t SumAbsoluteDifferencesU16(std::uint32_t a, std::uint32_t b) {
  std::uint32_t sum = 0;
  for (unsigned half_index = 0; half_index < 2; ++half_index) {
    const std::uint32_t lhs = (a >> (half_index * 16u)) & 0xffffu;
    const std::uint32_t rhs = (b >> (half_index * 16u)) & 0xffffu;
    sum += lhs >= rhs ? (lhs - rhs) : (rhs - lhs);
  }
  return sum;
}

std::uint32_t LerpU8(std::uint32_t a, std::uint32_t b, std::uint32_t round_mode) {
  std::uint32_t result = 0;
  for (unsigned byte_index = 0; byte_index < 4; ++byte_index) {
    const std::uint32_t lhs = (a >> (byte_index * 8u)) & 0xffu;
    const std::uint32_t rhs = (b >> (byte_index * 8u)) & 0xffu;
    const std::uint32_t round_bit = (round_mode >> (byte_index * 8u)) & 1u;
    const std::uint32_t average = (lhs + rhs + round_bit) >> 1u;
    result |= (average & 0xffu) << (byte_index * 8u);
  }
  return result;
}

std::uint32_t PermB32(std::uint32_t src0,
                      std::uint32_t src1,
                      std::uint32_t selectors) {
  std::uint32_t result = 0;
  for (unsigned byte_index = 0; byte_index < 4; ++byte_index) {
    const std::uint32_t selector = (selectors >> (byte_index * 8u)) & 0xffu;
    std::uint32_t value = 0;
    if (selector < 4u) {
      value = (src1 >> ((selector & 3u) * 8u)) & 0xffu;
    } else if (selector < 8u) {
      value = (src0 >> ((selector & 3u) * 8u)) & 0xffu;
    } else if (selector < 10u) {
      value = ((src1 >> ((selector & 1u) != 0u ? 31u : 15u)) & 1u) != 0u
                  ? 0xffu
                  : 0u;
    } else if (selector < 12u) {
      value = ((src0 >> ((selector & 1u) != 0u ? 31u : 15u)) & 1u) != 0u
                  ? 0xffu
                  : 0u;
    } else if (selector == 12u) {
      value = 0u;
    } else {
      value = 0xffu;
    }
    result |= value << (byte_index * 8u);
  }
  return result;
}

std::int32_t SignExtend24(std::uint32_t value) {
  value &= 0x00ffffffu;
  if ((value & 0x00800000u) != 0u) {
    value |= 0xff000000u;
  }
  return static_cast<std::int32_t>(value);
}

std::uint32_t MadI32I24(std::uint32_t src0,
                        std::uint32_t src1,
                        std::uint32_t src2) {
  const std::int64_t product = static_cast<std::int64_t>(SignExtend24(src0)) *
                               static_cast<std::int64_t>(SignExtend24(src1));
  const std::int64_t sum =
      product + static_cast<std::int64_t>(static_cast<std::int32_t>(src2));
  return static_cast<std::uint32_t>(sum);
}

std::uint32_t MadU32U24(std::uint32_t src0,
                        std::uint32_t src1,
                        std::uint32_t src2) {
  const std::uint64_t product =
      static_cast<std::uint64_t>(src0 & 0x00ffffffu) *
      static_cast<std::uint64_t>(src1 & 0x00ffffffu);
  return static_cast<std::uint32_t>(product + static_cast<std::uint64_t>(src2));
}

std::uint32_t CountLowBits(std::uint32_t value, std::uint32_t bit_count) {
  if (bit_count == 0u) {
    return 0u;
  }
  if (bit_count >= 32u) {
    return static_cast<std::uint32_t>(__builtin_popcount(value));
  }
  return static_cast<std::uint32_t>(
      __builtin_popcount(value & ((std::uint32_t{1} << bit_count) - 1u)));
}

bool ReadVectorPairOperandValue(const InstructionOperand& operand,
                                const WaveExecutionState& state,
                                std::size_t lane_index,
                                std::uint64_t* value,
                                std::string* error_message) {
  if (value == nullptr) {
    if (error_message != nullptr) {
      *error_message = "vector pair output must not be null";
    }
    return false;
  }

  if (operand.kind == OperandKind::kImm32) {
    *value = operand.imm32;
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }

  if (operand.kind == OperandKind::kVgpr) {
    if (operand.index + 1 >= state.vgprs.size()) {
      if (error_message != nullptr) {
        *error_message = "vector register pair out of range";
      }
      return false;
    }
    *value = static_cast<std::uint64_t>(state.vgprs[operand.index][lane_index]) |
             (static_cast<std::uint64_t>(
                  state.vgprs[operand.index + 1][lane_index])
              << 32);
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }

  if (operand.kind == OperandKind::kSgpr) {
    if (operand.index + 1 >= state.sgprs.size()) {
      if (error_message != nullptr) {
        *error_message = "scalar register pair out of range";
      }
      return false;
    }
    if (operand.index == kExecPairSgprIndex) {
      *value = state.exec_mask;
    } else {
      *value = static_cast<std::uint64_t>(state.sgprs[operand.index]) |
               (static_cast<std::uint64_t>(state.sgprs[operand.index + 1])
                << 32);
    }
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }

  if (error_message != nullptr) {
    *error_message = "unsupported 64-bit vector source operand";
  }
  return false;
}

bool WriteVectorPairOperandValue(const InstructionOperand& operand,
                                 std::size_t lane_index,
                                 std::uint64_t value,
                                 WaveExecutionState* state,
                                 std::string* error_message) {
  if (operand.kind != OperandKind::kVgpr) {
    if (error_message != nullptr) {
      *error_message = "expected vector destination pair operand";
    }
    return false;
  }
  if (operand.index + 1 >= state->vgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "vector destination pair out of range";
    }
    return false;
  }

  state->vgprs[operand.index][lane_index] = static_cast<std::uint32_t>(value);
  state->vgprs[operand.index + 1][lane_index] =
      static_cast<std::uint32_t>(value >> 32);
  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

std::uint8_t GetVectorMemoryRegisterDwordCount(std::string_view opcode) {
  if (opcode == "GLOBAL_LOAD_DWORDX4" || opcode == "GLOBAL_STORE_DWORDX4") {
    return 4;
  }
  if (opcode == "FLAT_LOAD_DWORDX4" || opcode == "FLAT_STORE_DWORDX4") {
    return 4;
  }
  if (opcode == "GLOBAL_LOAD_DWORDX3" || opcode == "GLOBAL_STORE_DWORDX3") {
    return 3;
  }
  if (opcode == "FLAT_LOAD_DWORDX3" || opcode == "FLAT_STORE_DWORDX3") {
    return 3;
  }
  if (opcode == "GLOBAL_LOAD_DWORDX2" || opcode == "GLOBAL_STORE_DWORDX2") {
    return 2;
  }
  if (opcode == "FLAT_LOAD_DWORDX2" || opcode == "FLAT_STORE_DWORDX2") {
    return 2;
  }
  return 1;
}

std::uint8_t GetVectorMemoryElementSizeBytes(std::string_view opcode) {
  if (opcode == "FLAT_LOAD_UBYTE" || opcode == "FLAT_LOAD_SBYTE" ||
      opcode == "FLAT_STORE_BYTE" || opcode == "GLOBAL_LOAD_UBYTE" ||
      opcode == "GLOBAL_LOAD_SBYTE" || opcode == "GLOBAL_STORE_BYTE") {
    return 1;
  }
  if (opcode == "FLAT_LOAD_USHORT" || opcode == "FLAT_LOAD_SSHORT" ||
      opcode == "FLAT_STORE_SHORT" || opcode == "GLOBAL_LOAD_USHORT" ||
      opcode == "GLOBAL_LOAD_SSHORT" || opcode == "GLOBAL_STORE_SHORT") {
    return 2;
  }
  return 4;
}

std::uint8_t GetGlobalAtomicDataDwordCount(std::string_view opcode) {
  if (opcode == "GLOBAL_ATOMIC_CMPSWAP_X2") {
    return 4;
  }
  if (opcode == "GLOBAL_ATOMIC_CMPSWAP") {
    return 2;
  }
  return GetGlobalAtomicMemoryDwordCount(opcode);
}

std::uint32_t AtomicIncU32(std::uint32_t old_value, std::uint32_t limit) {
  return old_value >= limit ? 0u : old_value + 1u;
}

std::uint32_t AtomicDecU32(std::uint32_t old_value, std::uint32_t limit) {
  return (old_value == 0u || old_value > limit) ? limit : old_value - 1u;
}

std::uint64_t AtomicIncU64(std::uint64_t old_value, std::uint64_t limit) {
  return old_value >= limit ? 0u : old_value + 1u;
}

std::uint64_t AtomicDecU64(std::uint64_t old_value, std::uint64_t limit) {
  return (old_value == 0u || old_value > limit) ? limit : old_value - 1u;
}

bool IsBranchOpcode(std::string_view opcode) {
  return opcode == "S_BRANCH" || opcode == "S_CBRANCH_SCC0" ||
         opcode == "S_CBRANCH_SCC1" || opcode == "S_CBRANCH_VCCZ" ||
         opcode == "S_CBRANCH_VCCNZ" || opcode == "S_CBRANCH_EXECZ" ||
         opcode == "S_CBRANCH_EXECNZ";
}

bool IsBarrierOpcode(std::string_view opcode) { return opcode == "S_BARRIER"; }

std::size_t FindLowestActiveLane(std::uint64_t exec_mask) {
  if (exec_mask == 0) {
    return WaveExecutionState::kLaneCount;
  }
#if defined(__GNUC__) || defined(__clang__)
  return static_cast<std::size_t>(__builtin_ctzll(exec_mask));
#else
  std::size_t lane_index = 0;
  while (((exec_mask >> lane_index) & 1ULL) == 0) {
    ++lane_index;
  }
  return lane_index;
#endif
}

std::size_t NormalizeWaveLaneIndex(std::uint32_t lane_selector) {
  return static_cast<std::size_t>(
      lane_selector & (WaveExecutionState::kLaneCount - 1));
}

bool ApplyExecMaskOpcode(std::string_view opcode,
                         std::uint64_t previous_exec,
                         std::uint64_t source,
                         std::uint64_t* next_exec) {
  if (next_exec == nullptr) {
    return false;
  }
  if (opcode == "S_AND_SAVEEXEC_B64") {
    *next_exec = previous_exec & source;
    return true;
  }
  if (opcode == "S_ANDN1_SAVEEXEC_B64") {
    *next_exec = previous_exec & ~source;
    return true;
  }
  if (opcode == "S_ANDN2_SAVEEXEC_B64") {
    *next_exec = source & ~previous_exec;
    return true;
  }
  if (opcode == "S_NAND_SAVEEXEC_B64") {
    *next_exec = ~(previous_exec & source);
    return true;
  }
  if (opcode == "S_OR_SAVEEXEC_B64") {
    *next_exec = previous_exec | source;
    return true;
  }
  if (opcode == "S_ORN1_SAVEEXEC_B64") {
    *next_exec = previous_exec | ~source;
    return true;
  }
  if (opcode == "S_ORN2_SAVEEXEC_B64") {
    *next_exec = source | ~previous_exec;
    return true;
  }
  if (opcode == "S_NOR_SAVEEXEC_B64") {
    *next_exec = ~(previous_exec | source);
    return true;
  }
  if (opcode == "S_XOR_SAVEEXEC_B64") {
    *next_exec = previous_exec ^ source;
    return true;
  }
  if (opcode == "S_XNOR_SAVEEXEC_B64") {
    *next_exec = ~(previous_exec ^ source);
    return true;
  }
  if (opcode == "S_ANDN1_WREXEC_B64") {
    *next_exec = previous_exec & ~source;
    return true;
  }
  if (opcode == "S_ANDN2_WREXEC_B64") {
    *next_exec = source & ~previous_exec;
    return true;
  }
  return false;
}

bool ApplyExecMaskOpcode(CompiledOpcode opcode,
                         std::uint64_t previous_exec,
                         std::uint64_t source,
                         std::uint64_t* next_exec) {
  if (next_exec == nullptr) {
    return false;
  }
  switch (opcode) {
    case CompiledOpcode::kSAndSaveexecB64:
      *next_exec = previous_exec & source;
      return true;
    case CompiledOpcode::kSAndn1SaveexecB64:
      *next_exec = previous_exec & ~source;
      return true;
    case CompiledOpcode::kSAndn2SaveexecB64:
      *next_exec = source & ~previous_exec;
      return true;
    case CompiledOpcode::kSNandSaveexecB64:
      *next_exec = ~(previous_exec & source);
      return true;
    case CompiledOpcode::kSOrSaveexecB64:
      *next_exec = previous_exec | source;
      return true;
    case CompiledOpcode::kSOrn1SaveexecB64:
      *next_exec = previous_exec | ~source;
      return true;
    case CompiledOpcode::kSOrn2SaveexecB64:
      *next_exec = source | ~previous_exec;
      return true;
    case CompiledOpcode::kSNorSaveexecB64:
      *next_exec = ~(previous_exec | source);
      return true;
    case CompiledOpcode::kSXorSaveexecB64:
      *next_exec = previous_exec ^ source;
      return true;
    case CompiledOpcode::kSXnorSaveexecB64:
      *next_exec = ~(previous_exec ^ source);
      return true;
    case CompiledOpcode::kSAndn1WrexecB64:
      *next_exec = previous_exec & ~source;
      return true;
    case CompiledOpcode::kSAndn2WrexecB64:
      *next_exec = source & ~previous_exec;
      return true;
    default:
      return false;
  }
}

std::uint32_t MakeBitfieldMask32(std::uint32_t width, std::uint32_t offset) {
  const std::uint32_t masked_width = width & 31u;
  const std::uint32_t masked_offset = offset & 31u;
  if (masked_width == 0) {
    return 0;
  }
  const std::uint32_t base_mask =
      static_cast<std::uint32_t>((std::uint64_t{1} << masked_width) - 1u);
  return base_mask << masked_offset;
}

std::uint64_t MakeBitfieldMask64(std::uint32_t width, std::uint32_t offset) {
  const std::uint32_t masked_width = width & 63u;
  const std::uint32_t masked_offset = offset & 63u;
  if (masked_width == 0) {
    return 0;
  }
  const std::uint64_t base_mask = (std::uint64_t{1} << masked_width) - 1u;
  return base_mask << masked_offset;
}

std::uint32_t ExtractUnsignedBitfield32(std::uint32_t source,
                                        std::uint32_t packed_field) {
  const std::uint32_t offset = packed_field & 31u;
  const std::uint32_t width = (packed_field >> 16) & 0x7fu;
  if (width == 0u || offset >= 32u) {
    return 0u;
  }
  const std::uint32_t effective_width = std::min(width, 32u - offset);
  if (effective_width >= 32u) {
    return source >> offset;
  }
  return (source >> offset) & ((std::uint32_t{1} << effective_width) - 1u);
}

std::uint32_t ExtractSignedBitfield32(std::uint32_t source,
                                      std::uint32_t packed_field) {
  const std::uint32_t offset = packed_field & 31u;
  const std::uint32_t width = (packed_field >> 16) & 0x7fu;
  if (width == 0u || offset >= 32u) {
    return 0u;
  }
  const std::uint32_t effective_width = std::min(width, 32u - offset);
  const std::uint32_t extracted =
      ExtractUnsignedBitfield32(source, packed_field);
  if (effective_width >= 32u) {
    return extracted;
  }
  const std::uint32_t sign_bit = std::uint32_t{1} << (effective_width - 1u);
  if ((extracted & sign_bit) == 0u) {
    return extracted;
  }
  return extracted | ~((std::uint32_t{1} << effective_width) - 1u);
}

std::uint64_t ExtractUnsignedBitfield64(std::uint64_t source,
                                        std::uint32_t packed_field) {
  const std::uint32_t offset = packed_field & 63u;
  const std::uint32_t width = (packed_field >> 16) & 0x7fu;
  if (width == 0u || offset >= 64u) {
    return 0u;
  }
  const std::uint32_t effective_width = std::min(width, 64u - offset);
  if (effective_width >= 64u) {
    return source >> offset;
  }
  return (source >> offset) & ((std::uint64_t{1} << effective_width) - 1u);
}

std::uint64_t ExtractSignedBitfield64(std::uint64_t source,
                                      std::uint32_t packed_field) {
  const std::uint32_t offset = packed_field & 63u;
  const std::uint32_t width = (packed_field >> 16) & 0x7fu;
  if (width == 0u || offset >= 64u) {
    return 0u;
  }
  const std::uint32_t effective_width = std::min(width, 64u - offset);
  const std::uint64_t extracted =
      ExtractUnsignedBitfield64(source, packed_field);
  if (effective_width >= 64u) {
    return extracted;
  }
  const std::uint64_t sign_bit = std::uint64_t{1} << (effective_width - 1u);
  if ((extracted & sign_bit) == 0u) {
    return extracted;
  }
  return extracted | ~((std::uint64_t{1} << effective_width) - 1u);
}

constexpr std::uint32_t kFpClassSignalingNaN = 0x001u;
constexpr std::uint32_t kFpClassQuietNaN = 0x002u;
constexpr std::uint32_t kFpClassNegativeInfinity = 0x004u;
constexpr std::uint32_t kFpClassNegativeNormal = 0x008u;
constexpr std::uint32_t kFpClassNegativeSubnormal = 0x010u;
constexpr std::uint32_t kFpClassNegativeZero = 0x020u;
constexpr std::uint32_t kFpClassPositiveZero = 0x040u;
constexpr std::uint32_t kFpClassPositiveSubnormal = 0x080u;
constexpr std::uint32_t kFpClassPositiveNormal = 0x100u;
constexpr std::uint32_t kFpClassPositiveInfinity = 0x200u;

std::uint32_t ClassifyFp16Mask(std::uint16_t bits) {
  const bool negative = (bits >> 15) != 0;
  const std::uint16_t exponent = static_cast<std::uint16_t>((bits >> 10) & 0x1fu);
  const std::uint16_t mantissa = static_cast<std::uint16_t>(bits & 0x03ffu);
  if (exponent == 0x1fu) {
    if (mantissa == 0u) {
      return negative ? kFpClassNegativeInfinity : kFpClassPositiveInfinity;
    }
    const bool quiet = (mantissa & 0x0200u) != 0u;
    return quiet ? kFpClassQuietNaN : kFpClassSignalingNaN;
  }
  if (exponent == 0u) {
    if (mantissa == 0u) {
      return negative ? kFpClassNegativeZero : kFpClassPositiveZero;
    }
    return negative ? kFpClassNegativeSubnormal : kFpClassPositiveSubnormal;
  }
  return negative ? kFpClassNegativeNormal : kFpClassPositiveNormal;
}

std::uint32_t ClassifyFp32Mask(std::uint32_t bits) {
  const bool negative = (bits >> 31) != 0;
  const std::uint32_t exponent = (bits >> 23) & 0xffu;
  const std::uint32_t mantissa = bits & 0x007fffffu;
  if (exponent == 0xffu) {
    if (mantissa == 0u) {
      return negative ? kFpClassNegativeInfinity : kFpClassPositiveInfinity;
    }
    const bool quiet = (mantissa & 0x00400000u) != 0u;
    return quiet ? kFpClassQuietNaN : kFpClassSignalingNaN;
  }
  if (exponent == 0u) {
    if (mantissa == 0u) {
      return negative ? kFpClassNegativeZero : kFpClassPositiveZero;
    }
    return negative ? kFpClassNegativeSubnormal : kFpClassPositiveSubnormal;
  }
  return negative ? kFpClassNegativeNormal : kFpClassPositiveNormal;
}

std::uint32_t ClassifyFp64Mask(std::uint64_t bits) {
  const bool negative = (bits >> 63) != 0;
  const std::uint32_t exponent = static_cast<std::uint32_t>((bits >> 52) & 0x7ffu);
  const std::uint64_t mantissa = bits & 0x000fffffffffffffULL;
  if (exponent == 0x7ffu) {
    if (mantissa == 0u) {
      return negative ? kFpClassNegativeInfinity : kFpClassPositiveInfinity;
    }
    const bool quiet = (mantissa & 0x0008000000000000ULL) != 0u;
    return quiet ? kFpClassQuietNaN : kFpClassSignalingNaN;
  }
  if (exponent == 0u) {
    if (mantissa == 0u) {
      return negative ? kFpClassNegativeZero : kFpClassPositiveZero;
    }
    return negative ? kFpClassNegativeSubnormal : kFpClassPositiveSubnormal;
  }
  return negative ? kFpClassNegativeNormal : kFpClassPositiveNormal;
}

bool EvaluateVectorCompareClassOpcode(std::string_view opcode,
                                      std::uint32_t value_bits,
                                      std::uint32_t class_mask,
                                      bool* result) {
  if (result == nullptr) {
    return false;
  }
  if (opcode == "V_CMP_CLASS_F16" || opcode == "V_CMPX_CLASS_F16") {
    *result =
        (ClassifyFp16Mask(static_cast<std::uint16_t>(value_bits)) & class_mask) !=
        0u;
    return true;
  }
  if (opcode == "V_CMP_CLASS_F32" || opcode == "V_CMPX_CLASS_F32") {
    *result = (ClassifyFp32Mask(value_bits) & class_mask) != 0u;
    return true;
  }
  return false;
}

bool EvaluateVectorCompareClassOpcode(std::string_view opcode,
                                      std::uint64_t value_bits,
                                      std::uint32_t class_mask,
                                      bool* result) {
  if (result == nullptr) {
    return false;
  }
  if (opcode != "V_CMP_CLASS_F64" && opcode != "V_CMPX_CLASS_F64") {
    return false;
  }
  *result = (ClassifyFp64Mask(value_bits) & class_mask) != 0u;
  return true;
}

bool EvaluateVectorCompareClassOpcode(CompiledOpcode opcode,
                                      std::uint32_t value_bits,
                                      std::uint32_t class_mask,
                                      bool* result) {
  if (result == nullptr) {
    return false;
  }
  switch (opcode) {
    case CompiledOpcode::kVCmpClassF16:
    case CompiledOpcode::kVCmpxClassF16:
      *result =
          (ClassifyFp16Mask(static_cast<std::uint16_t>(value_bits)) & class_mask) !=
          0u;
      return true;
    case CompiledOpcode::kVCmpClassF32:
    case CompiledOpcode::kVCmpxClassF32:
      *result = (ClassifyFp32Mask(value_bits) & class_mask) != 0u;
      return true;
    default:
      return false;
  }
}

bool EvaluateVectorCompareClassOpcode(CompiledOpcode opcode,
                                      std::uint64_t value_bits,
                                      std::uint32_t class_mask,
                                      bool* result) {
  if (result == nullptr) {
    return false;
  }
  switch (opcode) {
    case CompiledOpcode::kVCmpClassF64:
    case CompiledOpcode::kVCmpxClassF64:
      *result = (ClassifyFp64Mask(value_bits) & class_mask) != 0u;
      return true;
    default:
      return false;
  }
}

bool EvaluateVectorCompareOpcode(std::string_view opcode,
                                 std::uint32_t lhs,
                                 std::uint32_t rhs,
                                 bool* result) {
  if (result == nullptr) {
    return false;
  }
  if (opcode == "V_CMP_F_F16" || opcode == "V_CMP_LT_F16" ||
      opcode == "V_CMP_EQ_F16" || opcode == "V_CMP_LE_F16" ||
      opcode == "V_CMP_GT_F16" || opcode == "V_CMP_LG_F16" ||
      opcode == "V_CMP_GE_F16" || opcode == "V_CMP_O_F16" ||
      opcode == "V_CMP_U_F16" || opcode == "V_CMP_NGE_F16" ||
      opcode == "V_CMP_NLG_F16" || opcode == "V_CMP_NGT_F16" ||
      opcode == "V_CMP_NLE_F16" || opcode == "V_CMP_NEQ_F16" ||
      opcode == "V_CMP_NLT_F16" || opcode == "V_CMP_TRU_F16" ||
      opcode == "V_CMPX_F_F16" || opcode == "V_CMPX_LT_F16" ||
      opcode == "V_CMPX_EQ_F16" || opcode == "V_CMPX_LE_F16" ||
      opcode == "V_CMPX_GT_F16" || opcode == "V_CMPX_LG_F16" ||
      opcode == "V_CMPX_GE_F16" || opcode == "V_CMPX_O_F16" ||
      opcode == "V_CMPX_U_F16" || opcode == "V_CMPX_NGE_F16" ||
      opcode == "V_CMPX_NLG_F16" || opcode == "V_CMPX_NGT_F16" ||
      opcode == "V_CMPX_NLE_F16" || opcode == "V_CMPX_NEQ_F16" ||
      opcode == "V_CMPX_NLT_F16" || opcode == "V_CMPX_TRU_F16") {
    const float lhs_float = HalfToFloat(static_cast<std::uint16_t>(lhs));
    const float rhs_float = HalfToFloat(static_cast<std::uint16_t>(rhs));
    const bool unordered = std::isnan(lhs_float) || std::isnan(rhs_float);
    if (opcode == "V_CMP_F_F16" || opcode == "V_CMPX_F_F16") {
      *result = false;
      return true;
    }
    if (opcode == "V_CMP_LT_F16" || opcode == "V_CMPX_LT_F16") {
      *result = !unordered && lhs_float < rhs_float;
      return true;
    }
    if (opcode == "V_CMP_EQ_F16" || opcode == "V_CMPX_EQ_F16") {
      *result = !unordered && lhs_float == rhs_float;
      return true;
    }
    if (opcode == "V_CMP_LE_F16" || opcode == "V_CMPX_LE_F16") {
      *result = !unordered && lhs_float <= rhs_float;
      return true;
    }
    if (opcode == "V_CMP_GT_F16" || opcode == "V_CMPX_GT_F16") {
      *result = !unordered && lhs_float > rhs_float;
      return true;
    }
    if (opcode == "V_CMP_LG_F16" || opcode == "V_CMPX_LG_F16") {
      *result = !unordered && lhs_float != rhs_float;
      return true;
    }
    if (opcode == "V_CMP_GE_F16" || opcode == "V_CMPX_GE_F16") {
      *result = !unordered && lhs_float >= rhs_float;
      return true;
    }
    if (opcode == "V_CMP_O_F16" || opcode == "V_CMPX_O_F16") {
      *result = !unordered;
      return true;
    }
    if (opcode == "V_CMP_U_F16" || opcode == "V_CMPX_U_F16") {
      *result = unordered;
      return true;
    }
    if (opcode == "V_CMP_NGE_F16" || opcode == "V_CMPX_NGE_F16") {
      *result = unordered || lhs_float < rhs_float;
      return true;
    }
    if (opcode == "V_CMP_NLG_F16" || opcode == "V_CMPX_NLG_F16") {
      *result = unordered || lhs_float == rhs_float;
      return true;
    }
    if (opcode == "V_CMP_NGT_F16" || opcode == "V_CMPX_NGT_F16") {
      *result = unordered || lhs_float <= rhs_float;
      return true;
    }
    if (opcode == "V_CMP_NLE_F16" || opcode == "V_CMPX_NLE_F16") {
      *result = unordered || lhs_float > rhs_float;
      return true;
    }
    if (opcode == "V_CMP_NEQ_F16" || opcode == "V_CMPX_NEQ_F16") {
      *result = unordered || lhs_float != rhs_float;
      return true;
    }
    if (opcode == "V_CMP_NLT_F16" || opcode == "V_CMPX_NLT_F16") {
      *result = unordered || lhs_float >= rhs_float;
      return true;
    }
    if (opcode == "V_CMP_TRU_F16" || opcode == "V_CMPX_TRU_F16") {
      *result = true;
      return true;
    }
  }
  if (opcode == "V_CMP_F_F32" || opcode == "V_CMP_LT_F32" ||
      opcode == "V_CMP_EQ_F32" || opcode == "V_CMP_LE_F32" ||
      opcode == "V_CMP_GT_F32" || opcode == "V_CMP_LG_F32" ||
      opcode == "V_CMP_GE_F32" || opcode == "V_CMP_O_F32" ||
      opcode == "V_CMP_U_F32" || opcode == "V_CMP_NGE_F32" ||
      opcode == "V_CMP_NLG_F32" || opcode == "V_CMP_NGT_F32" ||
      opcode == "V_CMP_NLE_F32" || opcode == "V_CMP_NEQ_F32" ||
      opcode == "V_CMP_NLT_F32" || opcode == "V_CMP_TRU_F32" ||
      opcode == "V_CMPX_F_F32" || opcode == "V_CMPX_LT_F32" ||
      opcode == "V_CMPX_EQ_F32" || opcode == "V_CMPX_LE_F32" ||
      opcode == "V_CMPX_GT_F32" || opcode == "V_CMPX_LG_F32" ||
      opcode == "V_CMPX_GE_F32" || opcode == "V_CMPX_O_F32" ||
      opcode == "V_CMPX_U_F32" || opcode == "V_CMPX_NGE_F32" ||
      opcode == "V_CMPX_NLG_F32" || opcode == "V_CMPX_NGT_F32" ||
      opcode == "V_CMPX_NLE_F32" || opcode == "V_CMPX_NEQ_F32" ||
      opcode == "V_CMPX_NLT_F32" || opcode == "V_CMPX_TRU_F32") {
    const float lhs_float = BitCast<float>(lhs);
    const float rhs_float = BitCast<float>(rhs);
    const bool unordered = std::isnan(lhs_float) || std::isnan(rhs_float);
    if (opcode == "V_CMP_F_F32" || opcode == "V_CMPX_F_F32") {
      *result = false;
      return true;
    }
    if (opcode == "V_CMP_LT_F32" || opcode == "V_CMPX_LT_F32") {
      *result = !unordered && lhs_float < rhs_float;
      return true;
    }
    if (opcode == "V_CMP_EQ_F32" || opcode == "V_CMPX_EQ_F32") {
      *result = !unordered && lhs_float == rhs_float;
      return true;
    }
    if (opcode == "V_CMP_LE_F32" || opcode == "V_CMPX_LE_F32") {
      *result = !unordered && lhs_float <= rhs_float;
      return true;
    }
    if (opcode == "V_CMP_GT_F32" || opcode == "V_CMPX_GT_F32") {
      *result = !unordered && lhs_float > rhs_float;
      return true;
    }
    if (opcode == "V_CMP_LG_F32" || opcode == "V_CMPX_LG_F32") {
      *result = !unordered && lhs_float != rhs_float;
      return true;
    }
    if (opcode == "V_CMP_GE_F32" || opcode == "V_CMPX_GE_F32") {
      *result = !unordered && lhs_float >= rhs_float;
      return true;
    }
    if (opcode == "V_CMP_O_F32" || opcode == "V_CMPX_O_F32") {
      *result = !unordered;
      return true;
    }
    if (opcode == "V_CMP_U_F32" || opcode == "V_CMPX_U_F32") {
      *result = unordered;
      return true;
    }
    if (opcode == "V_CMP_NGE_F32" || opcode == "V_CMPX_NGE_F32") {
      *result = unordered || lhs_float < rhs_float;
      return true;
    }
    if (opcode == "V_CMP_NLG_F32" || opcode == "V_CMPX_NLG_F32") {
      *result = unordered || lhs_float == rhs_float;
      return true;
    }
    if (opcode == "V_CMP_NGT_F32" || opcode == "V_CMPX_NGT_F32") {
      *result = unordered || lhs_float <= rhs_float;
      return true;
    }
    if (opcode == "V_CMP_NLE_F32" || opcode == "V_CMPX_NLE_F32") {
      *result = unordered || lhs_float > rhs_float;
      return true;
    }
    if (opcode == "V_CMP_NEQ_F32" || opcode == "V_CMPX_NEQ_F32") {
      *result = unordered || lhs_float != rhs_float;
      return true;
    }
    if (opcode == "V_CMP_NLT_F32" || opcode == "V_CMPX_NLT_F32") {
      *result = unordered || lhs_float >= rhs_float;
      return true;
    }
    if (opcode == "V_CMP_TRU_F32" || opcode == "V_CMPX_TRU_F32") {
      *result = true;
      return true;
    }
  }
  if (opcode == "V_CMPX_F_I32" || opcode == "V_CMPX_F_U32") {
    *result = false;
    return true;
  }
  if (opcode == "V_CMPX_T_I32" || opcode == "V_CMPX_T_U32") {
    *result = true;
    return true;
  }
  if (opcode == "V_CMP_EQ_I32" || opcode == "V_CMPX_EQ_I32") {
    *result = BitCast<std::int32_t>(lhs) == BitCast<std::int32_t>(rhs);
    return true;
  }
  if (opcode == "V_CMP_NE_I32" || opcode == "V_CMPX_NE_I32") {
    *result = BitCast<std::int32_t>(lhs) != BitCast<std::int32_t>(rhs);
    return true;
  }
  if (opcode == "V_CMP_LT_I32" || opcode == "V_CMPX_LT_I32") {
    *result = BitCast<std::int32_t>(lhs) < BitCast<std::int32_t>(rhs);
    return true;
  }
  if (opcode == "V_CMP_LE_I32" || opcode == "V_CMPX_LE_I32") {
    *result = BitCast<std::int32_t>(lhs) <= BitCast<std::int32_t>(rhs);
    return true;
  }
  if (opcode == "V_CMP_GT_I32" || opcode == "V_CMPX_GT_I32") {
    *result = BitCast<std::int32_t>(lhs) > BitCast<std::int32_t>(rhs);
    return true;
  }
  if (opcode == "V_CMP_GE_I32" || opcode == "V_CMPX_GE_I32") {
    *result = BitCast<std::int32_t>(lhs) >= BitCast<std::int32_t>(rhs);
    return true;
  }
  if (opcode == "V_CMP_EQ_U32" || opcode == "V_CMPX_EQ_U32") {
    *result = lhs == rhs;
    return true;
  }
  if (opcode == "V_CMP_NE_U32" || opcode == "V_CMPX_NE_U32") {
    *result = lhs != rhs;
    return true;
  }
  if (opcode == "V_CMP_LT_U32" || opcode == "V_CMPX_LT_U32") {
    *result = lhs < rhs;
    return true;
  }
  if (opcode == "V_CMP_LE_U32" || opcode == "V_CMPX_LE_U32") {
    *result = lhs <= rhs;
    return true;
  }
  if (opcode == "V_CMP_GT_U32" || opcode == "V_CMPX_GT_U32") {
    *result = lhs > rhs;
    return true;
  }
  if (opcode == "V_CMP_GE_U32" || opcode == "V_CMPX_GE_U32") {
    *result = lhs >= rhs;
    return true;
  }
  return false;
}

bool EvaluateVectorCompareOpcode(std::string_view opcode,
                                 std::uint64_t lhs,
                                 std::uint64_t rhs,
                                 bool* result) {
  if (result == nullptr) {
    return false;
  }
  if (opcode == "V_CMP_F_F64" || opcode == "V_CMP_LT_F64" ||
      opcode == "V_CMP_EQ_F64" || opcode == "V_CMP_LE_F64" ||
      opcode == "V_CMP_GT_F64" || opcode == "V_CMP_LG_F64" ||
      opcode == "V_CMP_GE_F64" || opcode == "V_CMP_O_F64" ||
      opcode == "V_CMP_U_F64" || opcode == "V_CMP_NGE_F64" ||
      opcode == "V_CMP_NLG_F64" || opcode == "V_CMP_NGT_F64" ||
      opcode == "V_CMP_NLE_F64" || opcode == "V_CMP_NEQ_F64" ||
      opcode == "V_CMP_NLT_F64" || opcode == "V_CMP_TRU_F64" ||
      opcode == "V_CMPX_F_F64" || opcode == "V_CMPX_LT_F64" ||
      opcode == "V_CMPX_EQ_F64" || opcode == "V_CMPX_LE_F64" ||
      opcode == "V_CMPX_GT_F64" || opcode == "V_CMPX_LG_F64" ||
      opcode == "V_CMPX_GE_F64" || opcode == "V_CMPX_O_F64" ||
      opcode == "V_CMPX_U_F64" || opcode == "V_CMPX_NGE_F64" ||
      opcode == "V_CMPX_NLG_F64" || opcode == "V_CMPX_NGT_F64" ||
      opcode == "V_CMPX_NLE_F64" || opcode == "V_CMPX_NEQ_F64" ||
      opcode == "V_CMPX_NLT_F64" || opcode == "V_CMPX_TRU_F64") {
    const double lhs_double = BitCast<double>(lhs);
    const double rhs_double = BitCast<double>(rhs);
    const bool unordered = std::isnan(lhs_double) || std::isnan(rhs_double);
    if (opcode == "V_CMP_F_F64" || opcode == "V_CMPX_F_F64") {
      *result = false;
      return true;
    }
    if (opcode == "V_CMP_LT_F64" || opcode == "V_CMPX_LT_F64") {
      *result = !unordered && lhs_double < rhs_double;
      return true;
    }
    if (opcode == "V_CMP_EQ_F64" || opcode == "V_CMPX_EQ_F64") {
      *result = !unordered && lhs_double == rhs_double;
      return true;
    }
    if (opcode == "V_CMP_LE_F64" || opcode == "V_CMPX_LE_F64") {
      *result = !unordered && lhs_double <= rhs_double;
      return true;
    }
    if (opcode == "V_CMP_GT_F64" || opcode == "V_CMPX_GT_F64") {
      *result = !unordered && lhs_double > rhs_double;
      return true;
    }
    if (opcode == "V_CMP_LG_F64" || opcode == "V_CMPX_LG_F64") {
      *result = !unordered && lhs_double != rhs_double;
      return true;
    }
    if (opcode == "V_CMP_GE_F64" || opcode == "V_CMPX_GE_F64") {
      *result = !unordered && lhs_double >= rhs_double;
      return true;
    }
    if (opcode == "V_CMP_O_F64" || opcode == "V_CMPX_O_F64") {
      *result = !unordered;
      return true;
    }
    if (opcode == "V_CMP_U_F64" || opcode == "V_CMPX_U_F64") {
      *result = unordered;
      return true;
    }
    if (opcode == "V_CMP_NGE_F64" || opcode == "V_CMPX_NGE_F64") {
      *result = unordered || lhs_double < rhs_double;
      return true;
    }
    if (opcode == "V_CMP_NLG_F64" || opcode == "V_CMPX_NLG_F64") {
      *result = unordered || lhs_double == rhs_double;
      return true;
    }
    if (opcode == "V_CMP_NGT_F64" || opcode == "V_CMPX_NGT_F64") {
      *result = unordered || lhs_double <= rhs_double;
      return true;
    }
    if (opcode == "V_CMP_NLE_F64" || opcode == "V_CMPX_NLE_F64") {
      *result = unordered || lhs_double > rhs_double;
      return true;
    }
    if (opcode == "V_CMP_NEQ_F64" || opcode == "V_CMPX_NEQ_F64") {
      *result = unordered || lhs_double != rhs_double;
      return true;
    }
    if (opcode == "V_CMP_NLT_F64" || opcode == "V_CMPX_NLT_F64") {
      *result = unordered || lhs_double >= rhs_double;
      return true;
    }
    if (opcode == "V_CMP_TRU_F64" || opcode == "V_CMPX_TRU_F64") {
      *result = true;
      return true;
    }
  }
  if (opcode == "V_CMP_F_I64" || opcode == "V_CMP_F_U64" ||
      opcode == "V_CMPX_F_I64" || opcode == "V_CMPX_F_U64") {
    *result = false;
    return true;
  }
  if (opcode == "V_CMP_T_I64" || opcode == "V_CMP_T_U64" ||
      opcode == "V_CMPX_T_I64" || opcode == "V_CMPX_T_U64") {
    *result = true;
    return true;
  }
  if (opcode == "V_CMP_LT_I64" || opcode == "V_CMPX_LT_I64") {
    *result = BitCast<std::int64_t>(lhs) < BitCast<std::int64_t>(rhs);
    return true;
  }
  if (opcode == "V_CMP_EQ_I64" || opcode == "V_CMPX_EQ_I64") {
    *result = BitCast<std::int64_t>(lhs) == BitCast<std::int64_t>(rhs);
    return true;
  }
  if (opcode == "V_CMP_LE_I64" || opcode == "V_CMPX_LE_I64") {
    *result = BitCast<std::int64_t>(lhs) <= BitCast<std::int64_t>(rhs);
    return true;
  }
  if (opcode == "V_CMP_GT_I64" || opcode == "V_CMPX_GT_I64") {
    *result = BitCast<std::int64_t>(lhs) > BitCast<std::int64_t>(rhs);
    return true;
  }
  if (opcode == "V_CMP_NE_I64" || opcode == "V_CMPX_NE_I64") {
    *result = BitCast<std::int64_t>(lhs) != BitCast<std::int64_t>(rhs);
    return true;
  }
  if (opcode == "V_CMP_GE_I64" || opcode == "V_CMPX_GE_I64") {
    *result = BitCast<std::int64_t>(lhs) >= BitCast<std::int64_t>(rhs);
    return true;
  }
  if (opcode == "V_CMP_LT_U64" || opcode == "V_CMPX_LT_U64") {
    *result = lhs < rhs;
    return true;
  }
  if (opcode == "V_CMP_EQ_U64" || opcode == "V_CMPX_EQ_U64") {
    *result = lhs == rhs;
    return true;
  }
  if (opcode == "V_CMP_LE_U64" || opcode == "V_CMPX_LE_U64") {
    *result = lhs <= rhs;
    return true;
  }
  if (opcode == "V_CMP_GT_U64" || opcode == "V_CMPX_GT_U64") {
    *result = lhs > rhs;
    return true;
  }
  if (opcode == "V_CMP_NE_U64" || opcode == "V_CMPX_NE_U64") {
    *result = lhs != rhs;
    return true;
  }
  if (opcode == "V_CMP_GE_U64" || opcode == "V_CMPX_GE_U64") {
    *result = lhs >= rhs;
    return true;
  }
  return false;
}

bool EvaluateVectorCompareOpcode(CompiledOpcode opcode,
                                 std::uint32_t lhs,
                                 std::uint32_t rhs,
                                 bool* result) {
  if (result == nullptr) {
    return false;
  }
  const float lhs_half = HalfToFloat(static_cast<std::uint16_t>(lhs));
  const float rhs_half = HalfToFloat(static_cast<std::uint16_t>(rhs));
  const bool half_unordered = std::isnan(lhs_half) || std::isnan(rhs_half);
  const float lhs_float = BitCast<float>(lhs);
  const float rhs_float = BitCast<float>(rhs);
  const bool unordered = std::isnan(lhs_float) || std::isnan(rhs_float);
  switch (opcode) {
    case CompiledOpcode::kVCmpFF16:
    case CompiledOpcode::kVCmpxFF16:
      *result = false;
      return true;
    case CompiledOpcode::kVCmpLtF16:
    case CompiledOpcode::kVCmpxLtF16:
      *result = !half_unordered && lhs_half < rhs_half;
      return true;
    case CompiledOpcode::kVCmpEqF16:
    case CompiledOpcode::kVCmpxEqF16:
      *result = !half_unordered && lhs_half == rhs_half;
      return true;
    case CompiledOpcode::kVCmpLeF16:
    case CompiledOpcode::kVCmpxLeF16:
      *result = !half_unordered && lhs_half <= rhs_half;
      return true;
    case CompiledOpcode::kVCmpGtF16:
    case CompiledOpcode::kVCmpxGtF16:
      *result = !half_unordered && lhs_half > rhs_half;
      return true;
    case CompiledOpcode::kVCmpLgF16:
    case CompiledOpcode::kVCmpxLgF16:
      *result = !half_unordered && lhs_half != rhs_half;
      return true;
    case CompiledOpcode::kVCmpGeF16:
    case CompiledOpcode::kVCmpxGeF16:
      *result = !half_unordered && lhs_half >= rhs_half;
      return true;
    case CompiledOpcode::kVCmpOF16:
    case CompiledOpcode::kVCmpxOF16:
      *result = !half_unordered;
      return true;
    case CompiledOpcode::kVCmpUF16:
    case CompiledOpcode::kVCmpxUF16:
      *result = half_unordered;
      return true;
    case CompiledOpcode::kVCmpNgeF16:
    case CompiledOpcode::kVCmpxNgeF16:
      *result = half_unordered || lhs_half < rhs_half;
      return true;
    case CompiledOpcode::kVCmpNlgF16:
    case CompiledOpcode::kVCmpxNlgF16:
      *result = half_unordered || lhs_half == rhs_half;
      return true;
    case CompiledOpcode::kVCmpNgtF16:
    case CompiledOpcode::kVCmpxNgtF16:
      *result = half_unordered || lhs_half <= rhs_half;
      return true;
    case CompiledOpcode::kVCmpNleF16:
    case CompiledOpcode::kVCmpxNleF16:
      *result = half_unordered || lhs_half > rhs_half;
      return true;
    case CompiledOpcode::kVCmpNeqF16:
    case CompiledOpcode::kVCmpxNeqF16:
      *result = half_unordered || lhs_half != rhs_half;
      return true;
    case CompiledOpcode::kVCmpNltF16:
    case CompiledOpcode::kVCmpxNltF16:
      *result = half_unordered || lhs_half >= rhs_half;
      return true;
    case CompiledOpcode::kVCmpTruF16:
    case CompiledOpcode::kVCmpxTruF16:
      *result = true;
      return true;
    case CompiledOpcode::kVCmpFF32:
    case CompiledOpcode::kVCmpxFF32:
      *result = false;
      return true;
    case CompiledOpcode::kVCmpLtF32:
    case CompiledOpcode::kVCmpxLtF32:
      *result = !unordered && lhs_float < rhs_float;
      return true;
    case CompiledOpcode::kVCmpEqF32:
    case CompiledOpcode::kVCmpxEqF32:
      *result = !unordered && lhs_float == rhs_float;
      return true;
    case CompiledOpcode::kVCmpLeF32:
    case CompiledOpcode::kVCmpxLeF32:
      *result = !unordered && lhs_float <= rhs_float;
      return true;
    case CompiledOpcode::kVCmpGtF32:
    case CompiledOpcode::kVCmpxGtF32:
      *result = !unordered && lhs_float > rhs_float;
      return true;
    case CompiledOpcode::kVCmpLgF32:
    case CompiledOpcode::kVCmpxLgF32:
      *result = !unordered && lhs_float != rhs_float;
      return true;
    case CompiledOpcode::kVCmpGeF32:
    case CompiledOpcode::kVCmpxGeF32:
      *result = !unordered && lhs_float >= rhs_float;
      return true;
    case CompiledOpcode::kVCmpOF32:
    case CompiledOpcode::kVCmpxOF32:
      *result = !unordered;
      return true;
    case CompiledOpcode::kVCmpUF32:
    case CompiledOpcode::kVCmpxUF32:
      *result = unordered;
      return true;
    case CompiledOpcode::kVCmpNgeF32:
    case CompiledOpcode::kVCmpxNgeF32:
      *result = unordered || lhs_float < rhs_float;
      return true;
    case CompiledOpcode::kVCmpNlgF32:
    case CompiledOpcode::kVCmpxNlgF32:
      *result = unordered || lhs_float == rhs_float;
      return true;
    case CompiledOpcode::kVCmpNgtF32:
    case CompiledOpcode::kVCmpxNgtF32:
      *result = unordered || lhs_float <= rhs_float;
      return true;
    case CompiledOpcode::kVCmpNleF32:
    case CompiledOpcode::kVCmpxNleF32:
      *result = unordered || lhs_float > rhs_float;
      return true;
    case CompiledOpcode::kVCmpNeqF32:
    case CompiledOpcode::kVCmpxNeqF32:
      *result = unordered || lhs_float != rhs_float;
      return true;
    case CompiledOpcode::kVCmpNltF32:
    case CompiledOpcode::kVCmpxNltF32:
      *result = unordered || lhs_float >= rhs_float;
      return true;
    case CompiledOpcode::kVCmpTruF32:
    case CompiledOpcode::kVCmpxTruF32:
      *result = true;
      return true;
    case CompiledOpcode::kVCmpxFI32:
    case CompiledOpcode::kVCmpxFU32:
      *result = false;
      return true;
    case CompiledOpcode::kVCmpxTI32:
    case CompiledOpcode::kVCmpxTU32:
      *result = true;
      return true;
    case CompiledOpcode::kVCmpEqI32:
    case CompiledOpcode::kVCmpxEqI32:
      *result = BitCast<std::int32_t>(lhs) == BitCast<std::int32_t>(rhs);
      return true;
    case CompiledOpcode::kVCmpNeI32:
    case CompiledOpcode::kVCmpxNeI32:
      *result = BitCast<std::int32_t>(lhs) != BitCast<std::int32_t>(rhs);
      return true;
    case CompiledOpcode::kVCmpLtI32:
    case CompiledOpcode::kVCmpxLtI32:
      *result = BitCast<std::int32_t>(lhs) < BitCast<std::int32_t>(rhs);
      return true;
    case CompiledOpcode::kVCmpLeI32:
    case CompiledOpcode::kVCmpxLeI32:
      *result = BitCast<std::int32_t>(lhs) <= BitCast<std::int32_t>(rhs);
      return true;
    case CompiledOpcode::kVCmpGtI32:
    case CompiledOpcode::kVCmpxGtI32:
      *result = BitCast<std::int32_t>(lhs) > BitCast<std::int32_t>(rhs);
      return true;
    case CompiledOpcode::kVCmpGeI32:
    case CompiledOpcode::kVCmpxGeI32:
      *result = BitCast<std::int32_t>(lhs) >= BitCast<std::int32_t>(rhs);
      return true;
    case CompiledOpcode::kVCmpEqU32:
    case CompiledOpcode::kVCmpxEqU32:
      *result = lhs == rhs;
      return true;
    case CompiledOpcode::kVCmpNeU32:
    case CompiledOpcode::kVCmpxNeU32:
      *result = lhs != rhs;
      return true;
    case CompiledOpcode::kVCmpLtU32:
    case CompiledOpcode::kVCmpxLtU32:
      *result = lhs < rhs;
      return true;
    case CompiledOpcode::kVCmpLeU32:
    case CompiledOpcode::kVCmpxLeU32:
      *result = lhs <= rhs;
      return true;
    case CompiledOpcode::kVCmpGtU32:
    case CompiledOpcode::kVCmpxGtU32:
      *result = lhs > rhs;
      return true;
    case CompiledOpcode::kVCmpGeU32:
    case CompiledOpcode::kVCmpxGeU32:
      *result = lhs >= rhs;
      return true;
    default:
      return false;
  }
}

bool EvaluateVectorCompareOpcode(CompiledOpcode opcode,
                                 std::uint64_t lhs,
                                 std::uint64_t rhs,
                                 bool* result) {
  if (result == nullptr) {
    return false;
  }
  const double lhs_double = BitCast<double>(lhs);
  const double rhs_double = BitCast<double>(rhs);
  const bool unordered = std::isnan(lhs_double) || std::isnan(rhs_double);
  switch (opcode) {
    case CompiledOpcode::kVCmpFF64:
    case CompiledOpcode::kVCmpxFF64:
      *result = false;
      return true;
    case CompiledOpcode::kVCmpLtF64:
    case CompiledOpcode::kVCmpxLtF64:
      *result = !unordered && lhs_double < rhs_double;
      return true;
    case CompiledOpcode::kVCmpEqF64:
    case CompiledOpcode::kVCmpxEqF64:
      *result = !unordered && lhs_double == rhs_double;
      return true;
    case CompiledOpcode::kVCmpLeF64:
    case CompiledOpcode::kVCmpxLeF64:
      *result = !unordered && lhs_double <= rhs_double;
      return true;
    case CompiledOpcode::kVCmpGtF64:
    case CompiledOpcode::kVCmpxGtF64:
      *result = !unordered && lhs_double > rhs_double;
      return true;
    case CompiledOpcode::kVCmpLgF64:
    case CompiledOpcode::kVCmpxLgF64:
      *result = !unordered && lhs_double != rhs_double;
      return true;
    case CompiledOpcode::kVCmpGeF64:
    case CompiledOpcode::kVCmpxGeF64:
      *result = !unordered && lhs_double >= rhs_double;
      return true;
    case CompiledOpcode::kVCmpOF64:
    case CompiledOpcode::kVCmpxOF64:
      *result = !unordered;
      return true;
    case CompiledOpcode::kVCmpUF64:
    case CompiledOpcode::kVCmpxUF64:
      *result = unordered;
      return true;
    case CompiledOpcode::kVCmpNgeF64:
    case CompiledOpcode::kVCmpxNgeF64:
      *result = unordered || lhs_double < rhs_double;
      return true;
    case CompiledOpcode::kVCmpNlgF64:
    case CompiledOpcode::kVCmpxNlgF64:
      *result = unordered || lhs_double == rhs_double;
      return true;
    case CompiledOpcode::kVCmpNgtF64:
    case CompiledOpcode::kVCmpxNgtF64:
      *result = unordered || lhs_double <= rhs_double;
      return true;
    case CompiledOpcode::kVCmpNleF64:
    case CompiledOpcode::kVCmpxNleF64:
      *result = unordered || lhs_double > rhs_double;
      return true;
    case CompiledOpcode::kVCmpNeqF64:
    case CompiledOpcode::kVCmpxNeqF64:
      *result = unordered || lhs_double != rhs_double;
      return true;
    case CompiledOpcode::kVCmpNltF64:
    case CompiledOpcode::kVCmpxNltF64:
      *result = unordered || lhs_double >= rhs_double;
      return true;
    case CompiledOpcode::kVCmpTruF64:
    case CompiledOpcode::kVCmpxTruF64:
      *result = true;
      return true;
    case CompiledOpcode::kVCmpFI64:
    case CompiledOpcode::kVCmpFU64:
    case CompiledOpcode::kVCmpxFI64:
    case CompiledOpcode::kVCmpxFU64:
      *result = false;
      return true;
    case CompiledOpcode::kVCmpTI64:
    case CompiledOpcode::kVCmpTU64:
    case CompiledOpcode::kVCmpxTI64:
    case CompiledOpcode::kVCmpxTU64:
      *result = true;
      return true;
    case CompiledOpcode::kVCmpLtI64:
    case CompiledOpcode::kVCmpxLtI64:
      *result = BitCast<std::int64_t>(lhs) < BitCast<std::int64_t>(rhs);
      return true;
    case CompiledOpcode::kVCmpEqI64:
    case CompiledOpcode::kVCmpxEqI64:
      *result = BitCast<std::int64_t>(lhs) == BitCast<std::int64_t>(rhs);
      return true;
    case CompiledOpcode::kVCmpLeI64:
    case CompiledOpcode::kVCmpxLeI64:
      *result = BitCast<std::int64_t>(lhs) <= BitCast<std::int64_t>(rhs);
      return true;
    case CompiledOpcode::kVCmpGtI64:
    case CompiledOpcode::kVCmpxGtI64:
      *result = BitCast<std::int64_t>(lhs) > BitCast<std::int64_t>(rhs);
      return true;
    case CompiledOpcode::kVCmpNeI64:
    case CompiledOpcode::kVCmpxNeI64:
      *result = BitCast<std::int64_t>(lhs) != BitCast<std::int64_t>(rhs);
      return true;
    case CompiledOpcode::kVCmpGeI64:
    case CompiledOpcode::kVCmpxGeI64:
      *result = BitCast<std::int64_t>(lhs) >= BitCast<std::int64_t>(rhs);
      return true;
    case CompiledOpcode::kVCmpLtU64:
    case CompiledOpcode::kVCmpxLtU64:
      *result = lhs < rhs;
      return true;
    case CompiledOpcode::kVCmpEqU64:
    case CompiledOpcode::kVCmpxEqU64:
      *result = lhs == rhs;
      return true;
    case CompiledOpcode::kVCmpLeU64:
    case CompiledOpcode::kVCmpxLeU64:
      *result = lhs <= rhs;
      return true;
    case CompiledOpcode::kVCmpGtU64:
    case CompiledOpcode::kVCmpxGtU64:
      *result = lhs > rhs;
      return true;
    case CompiledOpcode::kVCmpNeU64:
    case CompiledOpcode::kVCmpxNeU64:
      *result = lhs != rhs;
      return true;
    case CompiledOpcode::kVCmpGeU64:
    case CompiledOpcode::kVCmpxGeU64:
      *result = lhs >= rhs;
      return true;
    default:
      return false;
  }
}

void SetVectorMemoryMetadata(CompiledInstruction* instruction,
                             CompiledOpcode opcode,
                             bool is_global,
                             bool is_load,
                             bool is_signed_load,
                             std::uint8_t register_dword_count,
                             std::uint8_t element_size_bytes) {
  instruction->opcode = opcode;
  instruction->flags =
      (is_global ? CompiledInstruction::kFlagIsGlobal : 0u) |
      (is_load ? CompiledInstruction::kFlagIsLoad : 0u) |
      (is_signed_load ? CompiledInstruction::kFlagIsSignedLoad : 0u);
  instruction->register_dword_count = register_dword_count;
  instruction->element_size_bytes = element_size_bytes;
}

void SetAtomicMetadata(CompiledInstruction* instruction,
                       CompiledOpcode opcode,
                       bool has_return,
                       std::uint8_t memory_dword_count,
                       std::uint8_t data_dword_count) {
  instruction->opcode = opcode;
  instruction->flags =
      CompiledInstruction::kFlagIsGlobal |
      (has_return ? CompiledInstruction::kFlagHasReturn : 0u);
  instruction->memory_dword_count = memory_dword_count;
  instruction->data_dword_count = data_dword_count;
}

bool TryCompileOpcode(std::string_view opcode,
                      CompiledInstruction* compiled_instruction) {
  if (compiled_instruction == nullptr) {
    return false;
  }

  if (opcode == "S_ENDPGM") {
    compiled_instruction->opcode = CompiledOpcode::kSEndpgm;
    return true;
  }
  if (opcode == "S_BARRIER") {
    compiled_instruction->opcode = CompiledOpcode::kSBarrier;
    return true;
  }
  if (opcode == "S_MOV_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSMovB32;
    return true;
  }
  if (opcode == "S_CMOVK_I32") {
    compiled_instruction->opcode = CompiledOpcode::kSCmovB32;
    return true;
  }
  if (opcode == "S_MOV_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSMovB64;
    return true;
  }
  if (opcode == "S_CMOV_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSCmovB32;
    return true;
  }
  if (opcode == "S_CMOV_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSCmovB64;
    return true;
  }
  if (opcode == "S_NOT_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSNotB32;
    return true;
  }
  if (opcode == "S_NOT_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSNotB64;
    return true;
  }
  if (opcode == "S_ABS_I32") {
    compiled_instruction->opcode = CompiledOpcode::kSAbsI32;
    return true;
  }
  if (opcode == "S_BREV_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSBrevB32;
    return true;
  }
  if (opcode == "S_BREV_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSBrevB64;
    return true;
  }
  if (opcode == "S_BCNT0_I32_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSBcnt0I32B32;
    return true;
  }
  if (opcode == "S_BCNT0_I32_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSBcnt0I32B64;
    return true;
  }
  if (opcode == "S_BCNT1_I32_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSBcnt1I32B32;
    return true;
  }
  if (opcode == "S_BCNT1_I32_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSBcnt1I32B64;
    return true;
  }
  if (opcode == "S_FF0_I32_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSFf0I32B32;
    return true;
  }
  if (opcode == "S_FF0_I32_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSFf0I32B64;
    return true;
  }
  if (opcode == "S_FF1_I32_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSFf1I32B32;
    return true;
  }
  if (opcode == "S_FF1_I32_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSFf1I32B64;
    return true;
  }
  if (opcode == "S_FLBIT_I32_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSFlbitI32B32;
    return true;
  }
  if (opcode == "S_FLBIT_I32_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSFlbitI32B64;
    return true;
  }
  if (opcode == "S_FLBIT_I32") {
    compiled_instruction->opcode = CompiledOpcode::kSFlbitI32;
    return true;
  }
  if (opcode == "S_FLBIT_I32_I64") {
    compiled_instruction->opcode = CompiledOpcode::kSFlbitI32I64;
    return true;
  }
  if (opcode == "S_BITREPLICATE_B64_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSBitreplicateB64B32;
    return true;
  }
  if (opcode == "S_QUADMASK_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSQuadmaskB32;
    return true;
  }
  if (opcode == "S_QUADMASK_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSQuadmaskB64;
    return true;
  }
  if (opcode == "S_SEXT_I32_I8") {
    compiled_instruction->opcode = CompiledOpcode::kSSextI32I8;
    return true;
  }
  if (opcode == "S_SEXT_I32_I16") {
    compiled_instruction->opcode = CompiledOpcode::kSSextI32I16;
    return true;
  }
  if (opcode == "S_BITSET0_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSBitset0B32;
    return true;
  }
  if (opcode == "S_BITSET0_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSBitset0B64;
    return true;
  }
  if (opcode == "S_BITSET1_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSBitset1B32;
    return true;
  }
  if (opcode == "S_BITSET1_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSBitset1B64;
    return true;
  }
  if (opcode == "S_AND_SAVEEXEC_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSAndSaveexecB64;
    return true;
  }
  if (opcode == "S_ANDN1_SAVEEXEC_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSAndn1SaveexecB64;
    return true;
  }
  if (opcode == "S_ANDN2_SAVEEXEC_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSAndn2SaveexecB64;
    return true;
  }
  if (opcode == "S_NAND_SAVEEXEC_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSNandSaveexecB64;
    return true;
  }
  if (opcode == "S_OR_SAVEEXEC_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSOrSaveexecB64;
    return true;
  }
  if (opcode == "S_ORN1_SAVEEXEC_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSOrn1SaveexecB64;
    return true;
  }
  if (opcode == "S_ORN2_SAVEEXEC_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSOrn2SaveexecB64;
    return true;
  }
  if (opcode == "S_NOR_SAVEEXEC_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSNorSaveexecB64;
    return true;
  }
  if (opcode == "S_XOR_SAVEEXEC_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSXorSaveexecB64;
    return true;
  }
  if (opcode == "S_XNOR_SAVEEXEC_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSXnorSaveexecB64;
    return true;
  }
  if (opcode == "S_ANDN1_WREXEC_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSAndn1WrexecB64;
    return true;
  }
  if (opcode == "S_ANDN2_WREXEC_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSAndn2WrexecB64;
    return true;
  }
  if (opcode == "V_NOP") {
    compiled_instruction->opcode = CompiledOpcode::kVNop;
    return true;
  }
  if (opcode == "V_MOV_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVMovB32;
    return true;
  }
  if (opcode == "V_MOV_B64") {
    compiled_instruction->opcode = CompiledOpcode::kVMovB64;
    return true;
  }
  if (opcode == "V_READFIRSTLANE_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVReadfirstlaneB32;
    return true;
  }
  if (opcode == "V_READLANE_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVReadlaneB32;
    return true;
  }
  if (opcode == "V_NOT_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVNotB32;
    return true;
  }
  if (opcode == "V_BFREV_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVBfrevB32;
    return true;
  }
  if (opcode == "V_FFBH_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVFfbhU32;
    return true;
  }
  if (opcode == "V_FFBL_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVFfblB32;
    return true;
  }
  if (opcode == "V_FFBH_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVFfbhI32;
    return true;
  }
  if (opcode == "V_CVT_F16_U16") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtF16U16;
    return true;
  }
  if (opcode == "V_CVT_F16_I16") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtF16I16;
    return true;
  }
  if (opcode == "V_CVT_U16_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtU16F16;
    return true;
  }
  if (opcode == "V_CVT_I16_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtI16F16;
    return true;
  }
  if (opcode == "V_SAT_PK_U8_I16") {
    compiled_instruction->opcode = CompiledOpcode::kVSatPkU8I16;
    return true;
  }
  if (opcode == "V_CVT_F32_UBYTE0") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtF32Ubyte0;
    return true;
  }
  if (opcode == "V_CVT_F32_UBYTE1") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtF32Ubyte1;
    return true;
  }
  if (opcode == "V_CVT_F32_UBYTE2") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtF32Ubyte2;
    return true;
  }
  if (opcode == "V_CVT_F32_UBYTE3") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtF32Ubyte3;
    return true;
  }
  if (opcode == "V_CVT_F32_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtF32I32;
    return true;
  }
  if (opcode == "V_CVT_F32_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtF32U32;
    return true;
  }
  if (opcode == "V_CVT_U32_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtU32F32;
    return true;
  }
  if (opcode == "V_CVT_I32_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtI32F32;
    return true;
  }
  if (opcode == "V_CVT_RPI_I32_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtRpiI32F32;
    return true;
  }
  if (opcode == "V_CVT_FLR_I32_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtFlrI32F32;
    return true;
  }
  if (opcode == "V_CVT_I32_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtI32F64;
    return true;
  }
  if (opcode == "V_CVT_U32_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtU32F64;
    return true;
  }
  if (opcode == "V_CVT_F16_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtF16F32;
    return true;
  }
  if (opcode == "V_CVT_F32_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtF32F16;
    return true;
  }
  if (opcode == "V_CVT_F32_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtF32F64;
    return true;
  }
  if (opcode == "V_CVT_F64_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtF64F32;
    return true;
  }
  if (opcode == "V_CVT_F64_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtF64I32;
    return true;
  }
  if (opcode == "V_CVT_F64_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVCvtF64U32;
    return true;
  }
  if (opcode == "V_RCP_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVRcpF16;
    return true;
  }
  if (opcode == "V_SQRT_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVSqrtF16;
    return true;
  }
  if (opcode == "V_RSQ_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVRsqF16;
    return true;
  }
  if (opcode == "V_LOG_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVLogF16;
    return true;
  }
  if (opcode == "V_EXP_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVExpF16;
    return true;
  }
  if (opcode == "V_SIN_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVSinF16;
    return true;
  }
  if (opcode == "V_COS_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCosF16;
    return true;
  }
  if (opcode == "V_FREXP_MANT_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVFrexpMantF16;
    return true;
  }
  if (opcode == "V_FREXP_EXP_I16_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVFrexpExpI16F16;
    return true;
  }
  if (opcode == "V_FRACT_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVFractF16;
    return true;
  }
  if (opcode == "V_TRUNC_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVTruncF16;
    return true;
  }
  if (opcode == "V_CEIL_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCeilF16;
    return true;
  }
  if (opcode == "V_RNDNE_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVRndneF16;
    return true;
  }
  if (opcode == "V_FLOOR_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVFloorF16;
    return true;
  }
  if (opcode == "V_EXP_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVExpF32;
    return true;
  }
  if (opcode == "V_EXP_LEGACY_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVExpLegacyF32;
    return true;
  }
  if (opcode == "V_LOG_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVLogF32;
    return true;
  }
  if (opcode == "V_LOG_LEGACY_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVLogLegacyF32;
    return true;
  }
  if (opcode == "V_RCP_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVRcpF32;
    return true;
  }
  if (opcode == "V_RCP_IFLAG_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVRcpIflagF32;
    return true;
  }
  if (opcode == "V_RSQ_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVRsqF32;
    return true;
  }
  if (opcode == "V_SQRT_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVSqrtF32;
    return true;
  }
  if (opcode == "V_SIN_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVSinF32;
    return true;
  }
  if (opcode == "V_COS_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCosF32;
    return true;
  }
  if (opcode == "V_FREXP_EXP_I32_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVFrexpExpI32F32;
    return true;
  }
  if (opcode == "V_FREXP_MANT_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVFrexpMantF32;
    return true;
  }
  if (opcode == "V_FRACT_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVFractF32;
    return true;
  }
  if (opcode == "V_TRUNC_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVTruncF32;
    return true;
  }
  if (opcode == "V_CEIL_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCeilF32;
    return true;
  }
  if (opcode == "V_RNDNE_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVRndneF32;
    return true;
  }
  if (opcode == "V_FLOOR_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVFloorF32;
    return true;
  }
  if (opcode == "V_RCP_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVRcpF64;
    return true;
  }
  if (opcode == "V_RSQ_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVRsqF64;
    return true;
  }
  if (opcode == "V_SQRT_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVSqrtF64;
    return true;
  }
  if (opcode == "V_FREXP_EXP_I32_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVFrexpExpI32F64;
    return true;
  }
  if (opcode == "V_FREXP_MANT_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVFrexpMantF64;
    return true;
  }
  if (opcode == "V_FRACT_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVFractF64;
    return true;
  }
  if (opcode == "V_TRUNC_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVTruncF64;
    return true;
  }
  if (opcode == "V_CEIL_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCeilF64;
    return true;
  }
  if (opcode == "V_RNDNE_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVRndneF64;
    return true;
  }
  if (opcode == "V_FLOOR_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVFloorF64;
    return true;
  }
  if (opcode == "V_MUL_LO_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVMulLoU32;
    return true;
  }
  if (opcode == "V_MUL_HI_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVMulHiU32;
    return true;
  }
  if (opcode == "V_MUL_HI_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVMulHiI32;
    return true;
  }
  if (opcode == "V_ADD_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVAddF32;
    return true;
  }
  if (opcode == "V_ADD_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVAddF16;
    return true;
  }
  if (opcode == "V_SUB_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVSubF32;
    return true;
  }
  if (opcode == "V_SUB_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVSubF16;
    return true;
  }
  if (opcode == "V_MUL_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVMulF32;
    return true;
  }
  if (opcode == "V_MUL_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVMulF16;
    return true;
  }
  if (opcode == "V_MIN_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVMinF32;
    return true;
  }
  if (opcode == "V_MIN_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVMinF16;
    return true;
  }
  if (opcode == "V_MAX_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVMaxF32;
    return true;
  }
  if (opcode == "V_MAX_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVMaxF16;
    return true;
  }
  if (opcode == "V_ADD_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVAddF64;
    return true;
  }
  if (opcode == "V_MUL_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVMulF64;
    return true;
  }
  if (opcode == "V_MIN_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVMinF64;
    return true;
  }
  if (opcode == "V_MAX_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVMaxF64;
    return true;
  }
  if (opcode == "V_BCNT_U32_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVBcntU32B32;
    return true;
  }
  if (opcode == "V_BFM_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVBfmB32;
    return true;
  }
  if (opcode == "V_MBCNT_LO_U32_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVMbcntLoU32B32;
    return true;
  }
  if (opcode == "V_MBCNT_HI_U32_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVMbcntHiU32B32;
    return true;
  }
  if (opcode == "V_LSHLREV_B64") {
    compiled_instruction->opcode = CompiledOpcode::kVLshlRevB64;
    return true;
  }
  if (opcode == "V_LSHRREV_B64") {
    compiled_instruction->opcode = CompiledOpcode::kVLshrRevB64;
    return true;
  }
  if (opcode == "V_ASHRREV_I64") {
    compiled_instruction->opcode = CompiledOpcode::kVAshrRevI64;
    return true;
  }
  if (opcode == "V_ADD3_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVAdd3U32;
    return true;
  }
  if (opcode == "V_LSHL_ADD_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVLshlAddU32;
    return true;
  }
  if (opcode == "V_LSHL_ADD_U64") {
    compiled_instruction->opcode = CompiledOpcode::kVLshlAddU64;
    return true;
  }
  if (opcode == "V_ADD_LSHL_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVAddLshlU32;
    return true;
  }
  if (opcode == "V_LSHL_OR_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVLshlOrB32;
    return true;
  }
  if (opcode == "V_AND_OR_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVAndOrB32;
    return true;
  }
  if (opcode == "V_OR3_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVOr3B32;
    return true;
  }
  if (opcode == "V_XAD_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVXadU32;
    return true;
  }
  if (opcode == "V_LERP_U8") {
    compiled_instruction->opcode = CompiledOpcode::kVLerpU8;
    return true;
  }
  if (opcode == "V_PERM_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVPermB32;
    return true;
  }
  if (opcode == "V_BFE_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVBfeU32;
    return true;
  }
  if (opcode == "V_BFE_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVBfeI32;
    return true;
  }
  if (opcode == "V_BFI_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVBfiB32;
    return true;
  }
  if (opcode == "V_ALIGNBIT_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVAlignbitB32;
    return true;
  }
  if (opcode == "V_ALIGNBYTE_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVAlignbyteB32;
    return true;
  }
  if (opcode == "V_MIN3_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVMin3I32;
    return true;
  }
  if (opcode == "V_MIN3_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVMin3U32;
    return true;
  }
  if (opcode == "V_MAX3_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVMax3I32;
    return true;
  }
  if (opcode == "V_MAX3_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVMax3U32;
    return true;
  }
  if (opcode == "V_MED3_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVMed3I32;
    return true;
  }
  if (opcode == "V_MED3_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVMed3U32;
    return true;
  }
  if (opcode == "V_SAD_U8") {
    compiled_instruction->opcode = CompiledOpcode::kVSadU8;
    return true;
  }
  if (opcode == "V_SAD_HI_U8") {
    compiled_instruction->opcode = CompiledOpcode::kVSadHiU8;
    return true;
  }
  if (opcode == "V_SAD_U16") {
    compiled_instruction->opcode = CompiledOpcode::kVSadU16;
    return true;
  }
  if (opcode == "V_SAD_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVSadU32;
    return true;
  }
  if (opcode == "V_FMA_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVFmaF32;
    return true;
  }
  if (opcode == "V_FMA_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVFmaF64;
    return true;
  }
  if (opcode == "V_MAD_I32_I24") {
    compiled_instruction->opcode = CompiledOpcode::kVMadI32I24;
    return true;
  }
  if (opcode == "V_MAD_U32_U24") {
    compiled_instruction->opcode = CompiledOpcode::kVMadU32U24;
    return true;
  }
  if (opcode == "V_MAD_U64_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVMadU64U32;
    return true;
  }
  if (opcode == "V_MAD_I64_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVMadI64I32;
    return true;
  }
  if (opcode == "V_CMP_F_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpFF16;
    return true;
  }
  if (opcode == "V_CMP_LT_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpLtF16;
    return true;
  }
  if (opcode == "V_CMP_EQ_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpEqF16;
    return true;
  }
  if (opcode == "V_CMP_LE_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpLeF16;
    return true;
  }
  if (opcode == "V_CMP_GT_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpGtF16;
    return true;
  }
  if (opcode == "V_CMP_LG_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpLgF16;
    return true;
  }
  if (opcode == "V_CMP_GE_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpGeF16;
    return true;
  }
  if (opcode == "V_CMP_O_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpOF16;
    return true;
  }
  if (opcode == "V_CMP_U_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpUF16;
    return true;
  }
  if (opcode == "V_CMP_NGE_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNgeF16;
    return true;
  }
  if (opcode == "V_CMP_NLG_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNlgF16;
    return true;
  }
  if (opcode == "V_CMP_NGT_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNgtF16;
    return true;
  }
  if (opcode == "V_CMP_NLE_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNleF16;
    return true;
  }
  if (opcode == "V_CMP_NEQ_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNeqF16;
    return true;
  }
  if (opcode == "V_CMP_NLT_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNltF16;
    return true;
  }
  if (opcode == "V_CMP_TRU_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpTruF16;
    return true;
  }
  if (opcode == "V_CMP_CLASS_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpClassF16;
    return true;
  }
  if (opcode == "V_CMP_F_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpFF32;
    return true;
  }
  if (opcode == "V_CMP_LT_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpLtF32;
    return true;
  }
  if (opcode == "V_CMP_EQ_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpEqF32;
    return true;
  }
  if (opcode == "V_CMP_LE_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpLeF32;
    return true;
  }
  if (opcode == "V_CMP_GT_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpGtF32;
    return true;
  }
  if (opcode == "V_CMP_LG_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpLgF32;
    return true;
  }
  if (opcode == "V_CMP_GE_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpGeF32;
    return true;
  }
  if (opcode == "V_CMP_O_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpOF32;
    return true;
  }
  if (opcode == "V_CMP_U_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpUF32;
    return true;
  }
  if (opcode == "V_CMP_NGE_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNgeF32;
    return true;
  }
  if (opcode == "V_CMP_NLG_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNlgF32;
    return true;
  }
  if (opcode == "V_CMP_NGT_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNgtF32;
    return true;
  }
  if (opcode == "V_CMP_NLE_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNleF32;
    return true;
  }
  if (opcode == "V_CMP_NEQ_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNeqF32;
    return true;
  }
  if (opcode == "V_CMP_NLT_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNltF32;
    return true;
  }
  if (opcode == "V_CMP_TRU_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpTruF32;
    return true;
  }
  if (opcode == "V_CMP_CLASS_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpClassF32;
    return true;
  }
  if (opcode == "V_CMP_F_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpFF64;
    return true;
  }
  if (opcode == "V_CMP_LT_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpLtF64;
    return true;
  }
  if (opcode == "V_CMP_EQ_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpEqF64;
    return true;
  }
  if (opcode == "V_CMP_LE_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpLeF64;
    return true;
  }
  if (opcode == "V_CMP_GT_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpGtF64;
    return true;
  }
  if (opcode == "V_CMP_LG_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpLgF64;
    return true;
  }
  if (opcode == "V_CMP_GE_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpGeF64;
    return true;
  }
  if (opcode == "V_CMP_O_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpOF64;
    return true;
  }
  if (opcode == "V_CMP_U_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpUF64;
    return true;
  }
  if (opcode == "V_CMP_NGE_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNgeF64;
    return true;
  }
  if (opcode == "V_CMP_NLG_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNlgF64;
    return true;
  }
  if (opcode == "V_CMP_NGT_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNgtF64;
    return true;
  }
  if (opcode == "V_CMP_NLE_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNleF64;
    return true;
  }
  if (opcode == "V_CMP_NEQ_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNeqF64;
    return true;
  }
  if (opcode == "V_CMP_NLT_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNltF64;
    return true;
  }
  if (opcode == "V_CMP_TRU_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpTruF64;
    return true;
  }
  if (opcode == "V_CMP_CLASS_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpClassF64;
    return true;
  }
  if (opcode == "V_CMPX_F_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxFF16;
    return true;
  }
  if (opcode == "V_CMPX_LT_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxLtF16;
    return true;
  }
  if (opcode == "V_CMPX_EQ_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxEqF16;
    return true;
  }
  if (opcode == "V_CMPX_LE_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxLeF16;
    return true;
  }
  if (opcode == "V_CMPX_GT_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxGtF16;
    return true;
  }
  if (opcode == "V_CMPX_LG_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxLgF16;
    return true;
  }
  if (opcode == "V_CMPX_GE_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxGeF16;
    return true;
  }
  if (opcode == "V_CMPX_O_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxOF16;
    return true;
  }
  if (opcode == "V_CMPX_U_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxUF16;
    return true;
  }
  if (opcode == "V_CMPX_NGE_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNgeF16;
    return true;
  }
  if (opcode == "V_CMPX_NLG_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNlgF16;
    return true;
  }
  if (opcode == "V_CMPX_NGT_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNgtF16;
    return true;
  }
  if (opcode == "V_CMPX_NLE_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNleF16;
    return true;
  }
  if (opcode == "V_CMPX_NEQ_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNeqF16;
    return true;
  }
  if (opcode == "V_CMPX_NLT_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNltF16;
    return true;
  }
  if (opcode == "V_CMPX_TRU_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxTruF16;
    return true;
  }
  if (opcode == "V_CMPX_CLASS_F16") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxClassF16;
    return true;
  }
  if (opcode == "V_CMPX_F_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxFF32;
    return true;
  }
  if (opcode == "V_CMPX_LT_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxLtF32;
    return true;
  }
  if (opcode == "V_CMPX_EQ_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxEqF32;
    return true;
  }
  if (opcode == "V_CMPX_LE_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxLeF32;
    return true;
  }
  if (opcode == "V_CMPX_GT_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxGtF32;
    return true;
  }
  if (opcode == "V_CMPX_LG_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxLgF32;
    return true;
  }
  if (opcode == "V_CMPX_GE_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxGeF32;
    return true;
  }
  if (opcode == "V_CMPX_O_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxOF32;
    return true;
  }
  if (opcode == "V_CMPX_U_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxUF32;
    return true;
  }
  if (opcode == "V_CMPX_NGE_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNgeF32;
    return true;
  }
  if (opcode == "V_CMPX_NLG_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNlgF32;
    return true;
  }
  if (opcode == "V_CMPX_NGT_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNgtF32;
    return true;
  }
  if (opcode == "V_CMPX_NLE_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNleF32;
    return true;
  }
  if (opcode == "V_CMPX_NEQ_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNeqF32;
    return true;
  }
  if (opcode == "V_CMPX_NLT_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNltF32;
    return true;
  }
  if (opcode == "V_CMPX_TRU_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxTruF32;
    return true;
  }
  if (opcode == "V_CMPX_CLASS_F32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxClassF32;
    return true;
  }
  if (opcode == "V_CMPX_F_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxFF64;
    return true;
  }
  if (opcode == "V_CMPX_LT_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxLtF64;
    return true;
  }
  if (opcode == "V_CMPX_EQ_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxEqF64;
    return true;
  }
  if (opcode == "V_CMPX_LE_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxLeF64;
    return true;
  }
  if (opcode == "V_CMPX_GT_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxGtF64;
    return true;
  }
  if (opcode == "V_CMPX_LG_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxLgF64;
    return true;
  }
  if (opcode == "V_CMPX_GE_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxGeF64;
    return true;
  }
  if (opcode == "V_CMPX_O_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxOF64;
    return true;
  }
  if (opcode == "V_CMPX_U_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxUF64;
    return true;
  }
  if (opcode == "V_CMPX_NGE_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNgeF64;
    return true;
  }
  if (opcode == "V_CMPX_NLG_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNlgF64;
    return true;
  }
  if (opcode == "V_CMPX_NGT_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNgtF64;
    return true;
  }
  if (opcode == "V_CMPX_NLE_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNleF64;
    return true;
  }
  if (opcode == "V_CMPX_NEQ_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNeqF64;
    return true;
  }
  if (opcode == "V_CMPX_NLT_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNltF64;
    return true;
  }
  if (opcode == "V_CMPX_TRU_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxTruF64;
    return true;
  }
  if (opcode == "V_CMPX_CLASS_F64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxClassF64;
    return true;
  }
  if (opcode == "V_CMP_EQ_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpEqI32;
    return true;
  }
  if (opcode == "V_CMP_NE_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNeI32;
    return true;
  }
  if (opcode == "V_CMP_LT_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpLtI32;
    return true;
  }
  if (opcode == "V_CMP_LE_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpLeI32;
    return true;
  }
  if (opcode == "V_CMP_GT_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpGtI32;
    return true;
  }
  if (opcode == "V_CMP_GE_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpGeI32;
    return true;
  }
  if (opcode == "V_CMP_EQ_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpEqU32;
    return true;
  }
  if (opcode == "V_CMP_NE_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNeU32;
    return true;
  }
  if (opcode == "V_CMP_LT_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpLtU32;
    return true;
  }
  if (opcode == "V_CMP_LE_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpLeU32;
    return true;
  }
  if (opcode == "V_CMP_GT_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpGtU32;
    return true;
  }
  if (opcode == "V_CMP_GE_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpGeU32;
    return true;
  }
  if (opcode == "V_CMPX_F_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxFI32;
    return true;
  }
  if (opcode == "V_CMPX_LT_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxLtI32;
    return true;
  }
  if (opcode == "V_CMPX_EQ_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxEqI32;
    return true;
  }
  if (opcode == "V_CMPX_LE_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxLeI32;
    return true;
  }
  if (opcode == "V_CMPX_GT_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxGtI32;
    return true;
  }
  if (opcode == "V_CMPX_NE_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNeI32;
    return true;
  }
  if (opcode == "V_CMPX_GE_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxGeI32;
    return true;
  }
  if (opcode == "V_CMPX_T_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxTI32;
    return true;
  }
  if (opcode == "V_CMPX_F_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxFU32;
    return true;
  }
  if (opcode == "V_CMPX_LT_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxLtU32;
    return true;
  }
  if (opcode == "V_CMPX_EQ_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxEqU32;
    return true;
  }
  if (opcode == "V_CMPX_LE_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxLeU32;
    return true;
  }
  if (opcode == "V_CMPX_GT_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxGtU32;
    return true;
  }
  if (opcode == "V_CMPX_NE_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNeU32;
    return true;
  }
  if (opcode == "V_CMPX_GE_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxGeU32;
    return true;
  }
  if (opcode == "V_CMPX_T_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxTU32;
    return true;
  }
  if (opcode == "V_CMP_F_I64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpFI64;
    return true;
  }
  if (opcode == "V_CMP_LT_I64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpLtI64;
    return true;
  }
  if (opcode == "V_CMP_EQ_I64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpEqI64;
    return true;
  }
  if (opcode == "V_CMP_LE_I64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpLeI64;
    return true;
  }
  if (opcode == "V_CMP_GT_I64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpGtI64;
    return true;
  }
  if (opcode == "V_CMP_NE_I64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNeI64;
    return true;
  }
  if (opcode == "V_CMP_GE_I64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpGeI64;
    return true;
  }
  if (opcode == "V_CMP_T_I64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpTI64;
    return true;
  }
  if (opcode == "V_CMP_F_U64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpFU64;
    return true;
  }
  if (opcode == "V_CMP_LT_U64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpLtU64;
    return true;
  }
  if (opcode == "V_CMP_EQ_U64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpEqU64;
    return true;
  }
  if (opcode == "V_CMP_LE_U64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpLeU64;
    return true;
  }
  if (opcode == "V_CMP_GT_U64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpGtU64;
    return true;
  }
  if (opcode == "V_CMP_NE_U64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpNeU64;
    return true;
  }
  if (opcode == "V_CMP_GE_U64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpGeU64;
    return true;
  }
  if (opcode == "V_CMP_T_U64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpTU64;
    return true;
  }
  if (opcode == "V_CMPX_F_I64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxFI64;
    return true;
  }
  if (opcode == "V_CMPX_LT_I64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxLtI64;
    return true;
  }
  if (opcode == "V_CMPX_EQ_I64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxEqI64;
    return true;
  }
  if (opcode == "V_CMPX_LE_I64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxLeI64;
    return true;
  }
  if (opcode == "V_CMPX_GT_I64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxGtI64;
    return true;
  }
  if (opcode == "V_CMPX_NE_I64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNeI64;
    return true;
  }
  if (opcode == "V_CMPX_GE_I64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxGeI64;
    return true;
  }
  if (opcode == "V_CMPX_T_I64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxTI64;
    return true;
  }
  if (opcode == "V_CMPX_F_U64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxFU64;
    return true;
  }
  if (opcode == "V_CMPX_LT_U64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxLtU64;
    return true;
  }
  if (opcode == "V_CMPX_EQ_U64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxEqU64;
    return true;
  }
  if (opcode == "V_CMPX_LE_U64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxLeU64;
    return true;
  }
  if (opcode == "V_CMPX_GT_U64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxGtU64;
    return true;
  }
  if (opcode == "V_CMPX_NE_U64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxNeU64;
    return true;
  }
  if (opcode == "V_CMPX_GE_U64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxGeU64;
    return true;
  }
  if (opcode == "V_CMPX_T_U64") {
    compiled_instruction->opcode = CompiledOpcode::kVCmpxTU64;
    return true;
  }
  if (opcode == "V_CNDMASK_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVCndmaskB32;
    return true;
  }
  if (opcode == "S_ADD_U32") {
    compiled_instruction->opcode = CompiledOpcode::kSAddU32;
    return true;
  }
  if (opcode == "S_ADD_I32") {
    compiled_instruction->opcode = CompiledOpcode::kSAddU32;
    return true;
  }
  if (opcode == "S_ADDK_I32") {
    compiled_instruction->opcode = CompiledOpcode::kSAddU32;
    return true;
  }
  if (opcode == "S_ADDC_U32") {
    compiled_instruction->opcode = CompiledOpcode::kSAddcU32;
    return true;
  }
  if (opcode == "S_SUB_U32") {
    compiled_instruction->opcode = CompiledOpcode::kSSubU32;
    return true;
  }
  if (opcode == "S_SUB_I32") {
    compiled_instruction->opcode = CompiledOpcode::kSSubU32;
    return true;
  }
  if (opcode == "S_SUBB_U32") {
    compiled_instruction->opcode = CompiledOpcode::kSSubbU32;
    return true;
  }
  if (opcode == "S_MIN_I32") {
    compiled_instruction->opcode = CompiledOpcode::kSMinI32;
    return true;
  }
  if (opcode == "S_MIN_U32") {
    compiled_instruction->opcode = CompiledOpcode::kSMinU32;
    return true;
  }
  if (opcode == "S_MAX_I32") {
    compiled_instruction->opcode = CompiledOpcode::kSMaxI32;
    return true;
  }
  if (opcode == "S_MAX_U32") {
    compiled_instruction->opcode = CompiledOpcode::kSMaxU32;
    return true;
  }
  if (opcode == "S_MUL_I32" || opcode == "S_MULK_I32") {
    compiled_instruction->opcode = CompiledOpcode::kSMulI32;
    return true;
  }
  if (opcode == "S_MUL_HI_U32") {
    compiled_instruction->opcode = CompiledOpcode::kSMulHiU32;
    return true;
  }
  if (opcode == "S_MUL_HI_I32") {
    compiled_instruction->opcode = CompiledOpcode::kSMulHiI32;
    return true;
  }
  if (opcode == "S_LSHL1_ADD_U32") {
    compiled_instruction->opcode = CompiledOpcode::kSLshl1AddU32;
    return true;
  }
  if (opcode == "S_LSHL2_ADD_U32") {
    compiled_instruction->opcode = CompiledOpcode::kSLshl2AddU32;
    return true;
  }
  if (opcode == "S_LSHL3_ADD_U32") {
    compiled_instruction->opcode = CompiledOpcode::kSLshl3AddU32;
    return true;
  }
  if (opcode == "S_LSHL4_ADD_U32") {
    compiled_instruction->opcode = CompiledOpcode::kSLshl4AddU32;
    return true;
  }
  if (opcode == "S_PACK_LL_B32_B16") {
    compiled_instruction->opcode = CompiledOpcode::kSPackLlB32B16;
    return true;
  }
  if (opcode == "S_PACK_LH_B32_B16") {
    compiled_instruction->opcode = CompiledOpcode::kSPackLhB32B16;
    return true;
  }
  if (opcode == "S_PACK_HH_B32_B16") {
    compiled_instruction->opcode = CompiledOpcode::kSPackHhB32B16;
    return true;
  }
  if (opcode == "S_CSELECT_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSCselectB32;
    return true;
  }
  if (opcode == "S_CSELECT_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSCselectB64;
    return true;
  }
  if (opcode == "S_ABSDIFF_I32") {
    compiled_instruction->opcode = CompiledOpcode::kSAbsdiffI32;
    return true;
  }
  if (opcode == "S_BFE_U32") {
    compiled_instruction->opcode = CompiledOpcode::kSBfeU32;
    return true;
  }
  if (opcode == "S_BFE_I32") {
    compiled_instruction->opcode = CompiledOpcode::kSBfeI32;
    return true;
  }
  if (opcode == "S_BFE_U64") {
    compiled_instruction->opcode = CompiledOpcode::kSBfeU64;
    return true;
  }
  if (opcode == "S_BFE_I64") {
    compiled_instruction->opcode = CompiledOpcode::kSBfeI64;
    return true;
  }
  if (opcode == "S_LSHL_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSLshlB32;
    return true;
  }
  if (opcode == "S_LSHL_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSLshlB64;
    return true;
  }
  if (opcode == "S_LSHR_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSLshrB32;
    return true;
  }
  if (opcode == "S_LSHR_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSLshrB64;
    return true;
  }
  if (opcode == "S_ASHR_I32") {
    compiled_instruction->opcode = CompiledOpcode::kSAshrI32;
    return true;
  }
  if (opcode == "S_ASHR_I64") {
    compiled_instruction->opcode = CompiledOpcode::kSAshrI64;
    return true;
  }
  if (opcode == "S_BFM_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSBfmB32;
    return true;
  }
  if (opcode == "S_BFM_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSBfmB64;
    return true;
  }
  if (opcode == "S_AND_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSAndB32;
    return true;
  }
  if (opcode == "S_ANDN2_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSAndn2B32;
    return true;
  }
  if (opcode == "S_NAND_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSNandB32;
    return true;
  }
  if (opcode == "S_AND_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSAndB64;
    return true;
  }
  if (opcode == "S_ANDN2_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSAndn2B64;
    return true;
  }
  if (opcode == "S_NAND_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSNandB64;
    return true;
  }
  if (opcode == "S_OR_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSOrB32;
    return true;
  }
  if (opcode == "S_ORN2_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSOrn2B32;
    return true;
  }
  if (opcode == "S_NOR_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSNorB32;
    return true;
  }
  if (opcode == "S_OR_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSOrB64;
    return true;
  }
  if (opcode == "S_ORN2_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSOrn2B64;
    return true;
  }
  if (opcode == "S_NOR_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSNorB64;
    return true;
  }
  if (opcode == "S_XOR_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSXorB32;
    return true;
  }
  if (opcode == "S_XNOR_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSXnorB32;
    return true;
  }
  if (opcode == "S_XOR_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSXorB64;
    return true;
  }
  if (opcode == "S_XNOR_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSXnorB64;
    return true;
  }
  if (opcode == "S_CMP_EQ_U32") {
    compiled_instruction->opcode = CompiledOpcode::kSCmpEqU32;
    return true;
  }
  if (opcode == "S_CMP_EQ_U64") {
    compiled_instruction->opcode = CompiledOpcode::kSCmpEqU64;
    return true;
  }
  if (opcode == "S_CMP_EQ_I32" || opcode == "S_CMPK_EQ_I32") {
    compiled_instruction->opcode = CompiledOpcode::kSCmpEqI32;
    return true;
  }
  if (opcode == "S_CMP_LG_U32") {
    compiled_instruction->opcode = CompiledOpcode::kSCmpLgU32;
    return true;
  }
  if (opcode == "S_CMP_LG_U64") {
    compiled_instruction->opcode = CompiledOpcode::kSCmpLgU64;
    return true;
  }
  if (opcode == "S_CMP_LG_I32" || opcode == "S_CMPK_LG_I32") {
    compiled_instruction->opcode = CompiledOpcode::kSCmpLgI32;
    return true;
  }
  if (opcode == "S_CMP_GT_I32" || opcode == "S_CMPK_GT_I32") {
    compiled_instruction->opcode = CompiledOpcode::kSCmpGtI32;
    return true;
  }
  if (opcode == "S_CMP_GE_I32" || opcode == "S_CMPK_GE_I32") {
    compiled_instruction->opcode = CompiledOpcode::kSCmpGeI32;
    return true;
  }
  if (opcode == "S_CMP_LT_I32" || opcode == "S_CMPK_LT_I32") {
    compiled_instruction->opcode = CompiledOpcode::kSCmpLtI32;
    return true;
  }
  if (opcode == "S_CMP_LE_I32" || opcode == "S_CMPK_LE_I32") {
    compiled_instruction->opcode = CompiledOpcode::kSCmpLeI32;
    return true;
  }
  if (opcode == "S_CMP_GT_U32" || opcode == "S_CMPK_GT_U32") {
    compiled_instruction->opcode = CompiledOpcode::kSCmpGtU32;
    return true;
  }
  if (opcode == "S_CMP_GE_U32" || opcode == "S_CMPK_GE_U32") {
    compiled_instruction->opcode = CompiledOpcode::kSCmpGeU32;
    return true;
  }
  if (opcode == "S_CMP_LT_U32" || opcode == "S_CMPK_LT_U32") {
    compiled_instruction->opcode = CompiledOpcode::kSCmpLtU32;
    return true;
  }
  if (opcode == "S_CMP_LE_U32" || opcode == "S_CMPK_LE_U32") {
    compiled_instruction->opcode = CompiledOpcode::kSCmpLeU32;
    return true;
  }
  if (opcode == "S_BITCMP0_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSBitcmp0B32;
    return true;
  }
  if (opcode == "S_BITCMP1_B32") {
    compiled_instruction->opcode = CompiledOpcode::kSBitcmp1B32;
    return true;
  }
  if (opcode == "S_BITCMP0_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSBitcmp0B64;
    return true;
  }
  if (opcode == "S_BITCMP1_B64") {
    compiled_instruction->opcode = CompiledOpcode::kSBitcmp1B64;
    return true;
  }
  if (opcode == "S_MOVK_I32") {
    compiled_instruction->opcode = CompiledOpcode::kSMovB32;
    return true;
  }
  if (opcode == "S_LOAD_DWORD") {
    compiled_instruction->opcode = CompiledOpcode::kSLoadDword;
    return true;
  }
  if (opcode == "S_LOAD_DWORDX2") {
    compiled_instruction->opcode = CompiledOpcode::kSLoadDwordX2;
    return true;
  }
  if (opcode == "S_STORE_DWORD") {
    compiled_instruction->opcode = CompiledOpcode::kSStoreDword;
    return true;
  }
  if (opcode == "V_ADD_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVAddU32;
    return true;
  }
  if (opcode == "V_ADD_CO_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVAddCoU32;
    return true;
  }
  if (opcode == "V_ADDC_CO_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVAddcCoU32;
    return true;
  }
  if (opcode == "V_SUB_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVSubU32;
    return true;
  }
  if (opcode == "V_SUB_CO_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVSubCoU32;
    return true;
  }
  if (opcode == "V_SUBB_CO_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVSubbCoU32;
    return true;
  }
  if (opcode == "V_SUBREV_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVSubRevU32;
    return true;
  }
  if (opcode == "V_SUBREV_CO_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVSubRevCoU32;
    return true;
  }
  if (opcode == "V_SUBBREV_CO_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVSubbrevCoU32;
    return true;
  }
  if (opcode == "V_MIN_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVMinI32;
    return true;
  }
  if (opcode == "V_MAX_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVMaxI32;
    return true;
  }
  if (opcode == "V_MIN_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVMinU32;
    return true;
  }
  if (opcode == "V_MAX_U32") {
    compiled_instruction->opcode = CompiledOpcode::kVMaxU32;
    return true;
  }
  if (opcode == "V_LSHRREV_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVLshrRevB32;
    return true;
  }
  if (opcode == "V_ASHRREV_I32") {
    compiled_instruction->opcode = CompiledOpcode::kVAshrRevI32;
    return true;
  }
  if (opcode == "V_LSHLREV_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVLshlRevB32;
    return true;
  }
  if (opcode == "V_WRITELANE_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVWritelaneB32;
    return true;
  }
  if (opcode == "V_AND_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVAndB32;
    return true;
  }
  if (opcode == "V_OR_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVOrB32;
    return true;
  }
  if (opcode == "V_XOR_B32") {
    compiled_instruction->opcode = CompiledOpcode::kVXorB32;
    return true;
  }
  if (opcode == "DS_NOP") {
    compiled_instruction->opcode = CompiledOpcode::kDsNop;
    return true;
  }
  if (opcode == "DS_WRITE_B32") {
    compiled_instruction->opcode = CompiledOpcode::kDsWriteB32;
    return true;
  }
  if (opcode == "DS_READ_B32") {
    compiled_instruction->opcode = CompiledOpcode::kDsReadB32;
    return true;
  }
  if (opcode == "DS_ADD_U32") {
    compiled_instruction->opcode = CompiledOpcode::kDsAddU32;
    return true;
  }
  if (opcode == "DS_SUB_U32") {
    compiled_instruction->opcode = CompiledOpcode::kDsSubU32;
    return true;
  }
  if (opcode == "DS_RSUB_U32") {
    compiled_instruction->opcode = CompiledOpcode::kDsRsubU32;
    return true;
  }
  if (opcode == "DS_INC_U32") {
    compiled_instruction->opcode = CompiledOpcode::kDsIncU32;
    return true;
  }
  if (opcode == "DS_DEC_U32") {
    compiled_instruction->opcode = CompiledOpcode::kDsDecU32;
    return true;
  }
  if (opcode == "DS_MIN_I32") {
    compiled_instruction->opcode = CompiledOpcode::kDsMinI32;
    return true;
  }
  if (opcode == "DS_MAX_I32") {
    compiled_instruction->opcode = CompiledOpcode::kDsMaxI32;
    return true;
  }
  if (opcode == "DS_MIN_U32") {
    compiled_instruction->opcode = CompiledOpcode::kDsMinU32;
    return true;
  }
  if (opcode == "DS_MAX_U32") {
    compiled_instruction->opcode = CompiledOpcode::kDsMaxU32;
    return true;
  }
  if (opcode == "DS_AND_B32") {
    compiled_instruction->opcode = CompiledOpcode::kDsAndB32;
    return true;
  }
  if (opcode == "DS_OR_B32") {
    compiled_instruction->opcode = CompiledOpcode::kDsOrB32;
    return true;
  }
  if (opcode == "DS_XOR_B32") {
    compiled_instruction->opcode = CompiledOpcode::kDsXorB32;
    return true;
  }
  if (opcode == "DS_MSKOR_B32") {
    compiled_instruction->opcode = CompiledOpcode::kDsMskorB32;
    return true;
  }
  if (opcode == "DS_CMPST_B32") {
    compiled_instruction->opcode = CompiledOpcode::kDsCmpstB32;
    return true;
  }
  if (opcode == "DS_CMPST_F32") {
    compiled_instruction->opcode = CompiledOpcode::kDsCmpstF32;
    return true;
  }
  if (opcode == "DS_ADD_F32") {
    compiled_instruction->opcode = CompiledOpcode::kDsAddF32;
    return true;
  }
  if (opcode == "DS_MIN_F32") {
    compiled_instruction->opcode = CompiledOpcode::kDsMinF32;
    return true;
  }
  if (opcode == "DS_MAX_F32") {
    compiled_instruction->opcode = CompiledOpcode::kDsMaxF32;
    return true;
  }
  if (opcode == "DS_WRITE_B8") {
    compiled_instruction->opcode = CompiledOpcode::kDsWriteB8;
    return true;
  }
  if (opcode == "DS_WRITE_B16") {
    compiled_instruction->opcode = CompiledOpcode::kDsWriteB16;
    return true;
  }
  if (opcode == "DS_WRITE2_B32") {
    compiled_instruction->opcode = CompiledOpcode::kDsWrite2B32;
    return true;
  }
  if (opcode == "DS_WRITE2ST64_B32") {
    compiled_instruction->opcode = CompiledOpcode::kDsWrite2St64B32;
    return true;
  }
  if (opcode == "DS_READ2_B32") {
    compiled_instruction->opcode = CompiledOpcode::kDsRead2B32;
    return true;
  }
  if (opcode == "DS_READ2ST64_B32") {
    compiled_instruction->opcode = CompiledOpcode::kDsRead2St64B32;
    return true;
  }
  if (opcode == "DS_READ_I8") {
    compiled_instruction->opcode = CompiledOpcode::kDsReadI8;
    return true;
  }
  if (opcode == "DS_READ_U8") {
    compiled_instruction->opcode = CompiledOpcode::kDsReadU8;
    return true;
  }
  if (opcode == "DS_READ_I16") {
    compiled_instruction->opcode = CompiledOpcode::kDsReadI16;
    return true;
  }
  if (opcode == "DS_READ_U16") {
    compiled_instruction->opcode = CompiledOpcode::kDsReadU16;
    return true;
  }
  if (opcode == "DS_ADD_RTN_U32") {
    compiled_instruction->opcode = CompiledOpcode::kDsAddRtnU32;
    return true;
  }
  if (opcode == "DS_SUB_RTN_U32") {
    compiled_instruction->opcode = CompiledOpcode::kDsSubRtnU32;
    return true;
  }
  if (opcode == "DS_RSUB_RTN_U32") {
    compiled_instruction->opcode = CompiledOpcode::kDsRsubRtnU32;
    return true;
  }
  if (opcode == "DS_INC_RTN_U32") {
    compiled_instruction->opcode = CompiledOpcode::kDsIncRtnU32;
    return true;
  }
  if (opcode == "DS_DEC_RTN_U32") {
    compiled_instruction->opcode = CompiledOpcode::kDsDecRtnU32;
    return true;
  }
  if (opcode == "DS_MIN_RTN_I32") {
    compiled_instruction->opcode = CompiledOpcode::kDsMinRtnI32;
    return true;
  }
  if (opcode == "DS_MAX_RTN_I32") {
    compiled_instruction->opcode = CompiledOpcode::kDsMaxRtnI32;
    return true;
  }
  if (opcode == "DS_MIN_RTN_U32") {
    compiled_instruction->opcode = CompiledOpcode::kDsMinRtnU32;
    return true;
  }
  if (opcode == "DS_MAX_RTN_U32") {
    compiled_instruction->opcode = CompiledOpcode::kDsMaxRtnU32;
    return true;
  }
  if (opcode == "DS_AND_RTN_B32") {
    compiled_instruction->opcode = CompiledOpcode::kDsAndRtnB32;
    return true;
  }
  if (opcode == "DS_OR_RTN_B32") {
    compiled_instruction->opcode = CompiledOpcode::kDsOrRtnB32;
    return true;
  }
  if (opcode == "DS_XOR_RTN_B32") {
    compiled_instruction->opcode = CompiledOpcode::kDsXorRtnB32;
    return true;
  }
  if (opcode == "DS_MSKOR_RTN_B32") {
    compiled_instruction->opcode = CompiledOpcode::kDsMskorRtnB32;
    return true;
  }
  if (opcode == "DS_WRXCHG_RTN_B32") {
    compiled_instruction->opcode = CompiledOpcode::kDsWrxchgRtnB32;
    return true;
  }
  if (opcode == "DS_CMPST_RTN_B32") {
    compiled_instruction->opcode = CompiledOpcode::kDsCmpstRtnB32;
    return true;
  }
  if (opcode == "DS_CMPST_RTN_F32") {
    compiled_instruction->opcode = CompiledOpcode::kDsCmpstRtnF32;
    return true;
  }
  if (opcode == "DS_WRAP_RTN_B32") {
    compiled_instruction->opcode = CompiledOpcode::kDsWrapRtnB32;
    return true;
  }
  if (opcode == "DS_ADD_RTN_F32") {
    compiled_instruction->opcode = CompiledOpcode::kDsAddRtnF32;
    return true;
  }
  if (opcode == "DS_MIN_RTN_F32") {
    compiled_instruction->opcode = CompiledOpcode::kDsMinRtnF32;
    return true;
  }
  if (opcode == "DS_MAX_RTN_F32") {
    compiled_instruction->opcode = CompiledOpcode::kDsMaxRtnF32;
    return true;
  }
  if (opcode == "S_BRANCH") {
    compiled_instruction->opcode = CompiledOpcode::kSBranch;
    return true;
  }
  if (opcode == "S_CBRANCH_SCC0") {
    compiled_instruction->opcode = CompiledOpcode::kSCbranchScc0;
    return true;
  }
  if (opcode == "S_CBRANCH_SCC1") {
    compiled_instruction->opcode = CompiledOpcode::kSCbranchScc1;
    return true;
  }
  if (opcode == "S_CBRANCH_VCCZ") {
    compiled_instruction->opcode = CompiledOpcode::kSCbranchVccz;
    return true;
  }
  if (opcode == "S_CBRANCH_VCCNZ") {
    compiled_instruction->opcode = CompiledOpcode::kSCbranchVccnz;
    return true;
  }
  if (opcode == "S_CBRANCH_EXECZ") {
    compiled_instruction->opcode = CompiledOpcode::kSCbranchExecz;
    return true;
  }
  if (opcode == "S_CBRANCH_EXECNZ") {
    compiled_instruction->opcode = CompiledOpcode::kSCbranchExecnz;
    return true;
  }

  if (opcode == "FLAT_LOAD_UBYTE") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kFlatLoadUByte,
                            false, true, false, 1, 1);
    return true;
  }
  if (opcode == "FLAT_LOAD_SBYTE") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kFlatLoadSByte,
                            false, true, true, 1, 1);
    return true;
  }
  if (opcode == "FLAT_LOAD_USHORT") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kFlatLoadUShort,
                            false, true, false, 1, 2);
    return true;
  }
  if (opcode == "FLAT_LOAD_SSHORT") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kFlatLoadSShort,
                            false, true, true, 1, 2);
    return true;
  }
  if (opcode == "FLAT_LOAD_DWORD") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kFlatLoadDword,
                            false, true, false, 1, 4);
    return true;
  }
  if (opcode == "FLAT_LOAD_DWORDX2") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kFlatLoadDwordX2,
                            false, true, false, 2, 4);
    return true;
  }
  if (opcode == "FLAT_LOAD_DWORDX3") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kFlatLoadDwordX3,
                            false, true, false, 3, 4);
    return true;
  }
  if (opcode == "FLAT_LOAD_DWORDX4") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kFlatLoadDwordX4,
                            false, true, false, 4, 4);
    return true;
  }
  if (opcode == "FLAT_STORE_BYTE") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kFlatStoreByte,
                            false, false, false, 1, 1);
    return true;
  }
  if (opcode == "FLAT_STORE_SHORT") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kFlatStoreShort,
                            false, false, false, 1, 2);
    return true;
  }
  if (opcode == "FLAT_STORE_DWORD") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kFlatStoreDword,
                            false, false, false, 1, 4);
    return true;
  }
  if (opcode == "FLAT_STORE_DWORDX2") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kFlatStoreDwordX2,
                            false, false, false, 2, 4);
    return true;
  }
  if (opcode == "FLAT_STORE_DWORDX3") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kFlatStoreDwordX3,
                            false, false, false, 3, 4);
    return true;
  }
  if (opcode == "FLAT_STORE_DWORDX4") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kFlatStoreDwordX4,
                            false, false, false, 4, 4);
    return true;
  }
  if (opcode == "GLOBAL_LOAD_UBYTE") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kGlobalLoadUByte,
                            true, true, false, 1, 1);
    return true;
  }
  if (opcode == "GLOBAL_LOAD_SBYTE") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kGlobalLoadSByte,
                            true, true, true, 1, 1);
    return true;
  }
  if (opcode == "GLOBAL_LOAD_USHORT") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kGlobalLoadUShort,
                            true, true, false, 1, 2);
    return true;
  }
  if (opcode == "GLOBAL_LOAD_SSHORT") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kGlobalLoadSShort,
                            true, true, true, 1, 2);
    return true;
  }
  if (opcode == "GLOBAL_LOAD_DWORD") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kGlobalLoadDword,
                            true, true, false, 1, 4);
    return true;
  }
  if (opcode == "GLOBAL_LOAD_DWORDX2") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kGlobalLoadDwordX2,
                            true, true, false, 2, 4);
    return true;
  }
  if (opcode == "GLOBAL_LOAD_DWORDX3") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kGlobalLoadDwordX3,
                            true, true, false, 3, 4);
    return true;
  }
  if (opcode == "GLOBAL_LOAD_DWORDX4") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kGlobalLoadDwordX4,
                            true, true, false, 4, 4);
    return true;
  }
  if (opcode == "GLOBAL_STORE_BYTE") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kGlobalStoreByte,
                            true, false, false, 1, 1);
    return true;
  }
  if (opcode == "GLOBAL_STORE_SHORT") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kGlobalStoreShort,
                            true, false, false, 1, 2);
    return true;
  }
  if (opcode == "GLOBAL_STORE_DWORD") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kGlobalStoreDword,
                            true, false, false, 1, 4);
    return true;
  }
  if (opcode == "GLOBAL_STORE_DWORDX2") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kGlobalStoreDwordX2,
                            true, false, false, 2, 4);
    return true;
  }
  if (opcode == "GLOBAL_STORE_DWORDX3") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kGlobalStoreDwordX3,
                            true, false, false, 3, 4);
    return true;
  }
  if (opcode == "GLOBAL_STORE_DWORDX4") {
    SetVectorMemoryMetadata(compiled_instruction, CompiledOpcode::kGlobalStoreDwordX4,
                            true, false, false, 4, 4);
    return true;
  }

  if (opcode == "GLOBAL_ATOMIC_SWAP") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicSwap,
                      true, 1, 1);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_CMPSWAP") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicCmpSwap,
                      true, 1, 2);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_ADD") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicAdd,
                      true, 1, 1);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_SUB") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicSub,
                      true, 1, 1);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_SMIN") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicSMin,
                      true, 1, 1);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_UMIN") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicUMin,
                      true, 1, 1);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_SMAX") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicSMax,
                      true, 1, 1);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_UMAX") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicUMax,
                      true, 1, 1);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_AND") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicAnd,
                      true, 1, 1);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_OR") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicOr,
                      true, 1, 1);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_XOR") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicXor,
                      true, 1, 1);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_INC") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicInc,
                      true, 1, 1);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_DEC") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicDec,
                      true, 1, 1);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_ADD_F32") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicAddF32,
                      true, 1, 1);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_PK_ADD_F16") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicPkAddF16,
                      true, 1, 1);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_ADD_F64") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicAddF64,
                      true, 2, 2);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_MIN_F64") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicMinF64,
                      true, 2, 2);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_MAX_F64") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicMaxF64,
                      true, 2, 2);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_PK_ADD_BF16") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicPkAddBf16,
                      true, 1, 1);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_SWAP_X2") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicSwapX2,
                      true, 2, 2);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_CMPSWAP_X2") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicCmpSwapX2,
                      true, 2, 4);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_ADD_X2") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicAddX2,
                      true, 2, 2);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_SUB_X2") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicSubX2,
                      true, 2, 2);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_SMIN_X2") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicSMinX2,
                      true, 2, 2);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_UMIN_X2") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicUMinX2,
                      true, 2, 2);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_SMAX_X2") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicSMaxX2,
                      true, 2, 2);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_UMAX_X2") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicUMaxX2,
                      true, 2, 2);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_AND_X2") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicAndX2,
                      true, 2, 2);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_OR_X2") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicOrX2,
                      true, 2, 2);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_XOR_X2") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicXorX2,
                      true, 2, 2);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_INC_X2") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicIncX2,
                      true, 2, 2);
    return true;
  }
  if (opcode == "GLOBAL_ATOMIC_DEC_X2") {
    SetAtomicMetadata(compiled_instruction, CompiledOpcode::kGlobalAtomicDecX2,
                      true, 2, 2);
    return true;
  }

  return false;
}

}  // namespace

bool Gfx950Interpreter::Supports(std::string_view opcode) const {
  if (FindGfx950Instruction(opcode) == nullptr) {
    return false;
  }

  return IsScalarMoveOpcode(opcode) || opcode == "S_ENDPGM" ||
         opcode == "V_NOP" || opcode == "DS_NOP" ||
         opcode == "V_MOV_B32" || opcode == "V_MOV_B64" ||
         IsExecMaskOpcode(opcode) || IsVectorToScalarOpcode(opcode) ||
         IsVectorUnaryOpcode(opcode) ||
         IsScalarBinaryOpcode(opcode) ||
         IsScalarCompareOpcode(opcode) || IsScalarMemoryOpcode(opcode) ||
         IsVectorBinaryOpcode(opcode) || IsVectorTernaryOpcode(opcode) ||
         IsVectorCompareOpcode(opcode) ||
         IsVectorMemoryOpcode(opcode) ||
         IsDsOpcode(opcode) || IsGlobalAtomicOpcode(opcode) ||
         IsBranchOpcode(opcode) || IsBarrierOpcode(opcode);
}

bool Gfx950Interpreter::CompileProgram(
    std::span<const DecodedInstruction> program,
    std::vector<CompiledInstruction>* compiled_program,
    std::string* error_message) const {
  if (compiled_program == nullptr) {
    if (error_message != nullptr) {
      *error_message = "compiled program output must not be null";
    }
    return false;
  }

  compiled_program->clear();
  compiled_program->reserve(program.size());
  for (const DecodedInstruction& instruction : program) {
    CompiledInstruction compiled_instruction;
    if (!CompileInstruction(instruction, &compiled_instruction, error_message)) {
      return false;
    }
    compiled_program->push_back(compiled_instruction);
  }
  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool Gfx950Interpreter::CompileInstruction(
    const DecodedInstruction& instruction,
    CompiledInstruction* compiled_instruction,
    std::string* error_message) const {
  if (compiled_instruction == nullptr) {
    if (error_message != nullptr) {
      *error_message = "compiled instruction output must not be null";
    }
    return false;
  }
  if (FindGfx950Instruction(instruction.opcode) == nullptr) {
    if (error_message != nullptr) {
      *error_message = "unknown gfx950 opcode";
    }
    return false;
  }
  if (!TryCompileOpcode(instruction.opcode, compiled_instruction)) {
    if (error_message != nullptr) {
      *error_message =
          "opcode exists in catalog but execution semantics are not implemented";
    }
    return false;
  }

  compiled_instruction->operands = instruction.operands;
  compiled_instruction->operand_count = instruction.operand_count;
  if ((compiled_instruction->flags & CompiledInstruction::kFlagIsGlobal) != 0 &&
      compiled_instruction->memory_dword_count != 0) {
    compiled_instruction->flags &= ~CompiledInstruction::kFlagHasReturn;
    if (instruction.operand_count == 5) {
      compiled_instruction->flags |= CompiledInstruction::kFlagHasReturn;
    }
  }
  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool Gfx950Interpreter::ExecuteProgram(std::span<const DecodedInstruction> program,
                                       WaveExecutionState* state,
                                       std::string* error_message) const {
  return ExecuteProgram(program, state, nullptr, error_message);
}

bool Gfx950Interpreter::ExecuteProgram(std::span<const DecodedInstruction> program,
                                       WaveExecutionState* state,
                                       ExecutionMemory* memory,
                                       std::string* error_message) const {
  ProgramRunState run_state = ProgramRunState::kCompleted;
  if (!ExecuteProgramUntilYield(program, state, memory, nullptr, &run_state,
                                error_message)) {
    return false;
  }
  if (run_state != ProgramRunState::kCompleted) {
    if (error_message != nullptr) {
      *error_message = "program yielded before completion";
    }
    return false;
  }
  return true;
}

bool Gfx950Interpreter::ExecuteProgram(std::span<const CompiledInstruction> program,
                                       WaveExecutionState* state,
                                       std::string* error_message) const {
  return ExecuteProgram(program, state, nullptr, error_message);
}

bool Gfx950Interpreter::ExecuteProgram(std::span<const CompiledInstruction> program,
                                       WaveExecutionState* state,
                                       ExecutionMemory* memory,
                                       std::string* error_message) const {
  ProgramRunState run_state = ProgramRunState::kCompleted;
  if (!ExecuteProgramUntilYield(program, state, memory, nullptr, &run_state,
                                error_message)) {
    return false;
  }
  if (run_state != ProgramRunState::kCompleted) {
    if (error_message != nullptr) {
      *error_message = "program yielded before completion";
    }
    return false;
  }
  return true;
}

bool Gfx950Interpreter::ExecuteProgramUntilYield(
    std::span<const DecodedInstruction> program,
    WaveExecutionState* state,
    ExecutionMemory* memory,
    const WorkgroupExecutionContext* workgroup,
    ProgramRunState* run_state,
    std::string* error_message) const {
  if (state == nullptr) {
    if (error_message != nullptr) {
      *error_message = "execution state must not be null";
    }
    return false;
  }

  if (run_state != nullptr) {
    *run_state = ProgramRunState::kCompleted;
  }
  if (state->pc == 0) {
    state->halted = false;
    state->waiting_on_barrier = false;
  }
  while (!state->halted && state->pc < program.size()) {
    const std::uint64_t next_pc = state->pc + 1;
    const DecodedInstruction& instruction =
        program[static_cast<std::size_t>(state->pc)];
    bool pc_was_updated = false;
    bool wave_yielded = false;
    if (!ExecuteInstruction(instruction, state, memory, workgroup, &pc_was_updated,
                            &wave_yielded,
                            error_message)) {
      return false;
    }
    if (!state->halted && !pc_was_updated) {
      state->pc = next_pc;
    }
    if (wave_yielded) {
      if (run_state != nullptr) {
        *run_state = ProgramRunState::kBlockedOnBarrier;
      }
      return true;
    }
  }
  return true;
}

bool Gfx950Interpreter::ExecuteProgramUntilYield(
    std::span<const CompiledInstruction> program,
    WaveExecutionState* state,
    ExecutionMemory* memory,
    const WorkgroupExecutionContext* workgroup,
    ProgramRunState* run_state,
    std::string* error_message) const {
  if (state == nullptr) {
    if (error_message != nullptr) {
      *error_message = "execution state must not be null";
    }
    return false;
  }

  if (run_state != nullptr) {
    *run_state = ProgramRunState::kCompleted;
  }
  if (state->pc == 0) {
    state->halted = false;
    state->waiting_on_barrier = false;
  }
  while (!state->halted && state->pc < program.size()) {
    const std::uint64_t next_pc = state->pc + 1;
    const CompiledInstruction& instruction =
        program[static_cast<std::size_t>(state->pc)];
    bool pc_was_updated = false;
    bool wave_yielded = false;
    if (!ExecuteInstruction(instruction, state, memory, workgroup, &pc_was_updated,
                            &wave_yielded, error_message)) {
      return false;
    }
    if (!state->halted && !pc_was_updated) {
      state->pc = next_pc;
    }
    if (wave_yielded) {
      if (run_state != nullptr) {
        *run_state = ProgramRunState::kBlockedOnBarrier;
      }
      return true;
    }
  }
  return true;
}

bool Gfx950Interpreter::ExecuteInstruction(const DecodedInstruction& instruction,
                                           WaveExecutionState* state,
                                           ExecutionMemory* memory,
                                           const WorkgroupExecutionContext* workgroup,
                                           bool* pc_was_updated,
                                           bool* wave_yielded,
                                           std::string* error_message) const {
  if (pc_was_updated != nullptr) {
    *pc_was_updated = false;
  }
  if (wave_yielded != nullptr) {
    *wave_yielded = false;
  }
  if (FindGfx950Instruction(instruction.opcode) == nullptr) {
    if (error_message != nullptr) {
      *error_message = "unknown gfx950 opcode";
    }
    return false;
  }

  if (instruction.opcode == "S_ENDPGM") {
    if (!ValidateOperandCount(instruction, 0, error_message)) {
      return false;
    }
    state->halted = true;
    state->waiting_on_barrier = false;
    return true;
  }

  if (instruction.opcode == "V_NOP") {
    return ValidateOperandCount(instruction, 0, error_message);
  }

  if (instruction.opcode == "DS_NOP") {
    return ValidateOperandCount(instruction, 0, error_message);
  }

  if (IsBarrierOpcode(instruction.opcode)) {
    return ExecuteBarrier(instruction, state, workgroup, wave_yielded,
                          error_message);
  }
  if (IsScalarMoveOpcode(instruction.opcode)) {
    return ExecuteScalarMove(instruction, state, error_message);
  }
  if (IsExecMaskOpcode(instruction.opcode)) {
    return ExecuteExecMaskOp(instruction, state, error_message);
  }
  if (instruction.opcode == "V_MOV_B32" || instruction.opcode == "V_MOV_B64") {
    return ExecuteVectorMove(instruction, state, error_message);
  }
  if (IsVectorToScalarOpcode(instruction.opcode)) {
    return ExecuteVectorToScalar(instruction, state, error_message);
  }
  if (IsVectorUnaryOpcode(instruction.opcode)) {
    return ExecuteVectorUnary(instruction, state, error_message);
  }
  if (IsScalarBinaryOpcode(instruction.opcode)) {
    return ExecuteScalarBinary(instruction, state, error_message);
  }
  if (IsScalarCompareOpcode(instruction.opcode)) {
    return ExecuteScalarCompare(instruction, state, error_message);
  }
  if (IsScalarMemoryOpcode(instruction.opcode)) {
    return ExecuteScalarMemory(instruction, state, memory, error_message);
  }
  if (IsVectorBinaryOpcode(instruction.opcode)) {
    return ExecuteVectorBinary(instruction, state, error_message);
  }
  if (IsVectorTernaryOpcode(instruction.opcode)) {
    return ExecuteVectorTernary(instruction, state, error_message);
  }
  if (IsVectorCompareOpcode(instruction.opcode)) {
    return ExecuteVectorCompare(instruction, state, error_message);
  }
  if (IsVectorMemoryOpcode(instruction.opcode)) {
    return ExecuteVectorMemory(instruction, state, memory, error_message);
  }
  if (IsDsOpcode(instruction.opcode)) {
    return ExecuteDsMemory(instruction, state, workgroup, error_message);
  }
  if (IsGlobalAtomicOpcode(instruction.opcode)) {
    return ExecuteGlobalAtomic(instruction, state, memory, error_message);
  }
  if (IsBranchOpcode(instruction.opcode)) {
    return ExecuteBranch(instruction, state, pc_was_updated, error_message);
  }

  if (error_message != nullptr) {
    *error_message = "opcode exists in catalog but execution semantics are not implemented";
  }
  return false;
}

bool Gfx950Interpreter::ExecuteInstruction(const CompiledInstruction& instruction,
                                           WaveExecutionState* state,
                                           ExecutionMemory* memory,
                                           const WorkgroupExecutionContext* workgroup,
                                           bool* pc_was_updated,
                                           bool* wave_yielded,
                                           std::string* error_message) const {
  if (pc_was_updated != nullptr) {
    *pc_was_updated = false;
  }
  if (wave_yielded != nullptr) {
    *wave_yielded = false;
  }

  switch (instruction.opcode) {
    case CompiledOpcode::kSEndpgm:
      if (!ValidateOperandCount(instruction, 0, error_message)) {
        return false;
      }
      state->halted = true;
      state->waiting_on_barrier = false;
      return true;
    case CompiledOpcode::kVNop:
      return ValidateOperandCount(instruction, 0, error_message);
    case CompiledOpcode::kSBarrier:
      return ExecuteBarrier(instruction, state, workgroup, wave_yielded,
                            error_message);
    case CompiledOpcode::kSMovB32:
    case CompiledOpcode::kSMovB64:
    case CompiledOpcode::kSCmovB32:
    case CompiledOpcode::kSCmovB64:
    case CompiledOpcode::kSNotB32:
    case CompiledOpcode::kSNotB64:
    case CompiledOpcode::kSAbsI32:
    case CompiledOpcode::kSBrevB32:
    case CompiledOpcode::kSBrevB64:
    case CompiledOpcode::kSBcnt0I32B32:
    case CompiledOpcode::kSBcnt0I32B64:
    case CompiledOpcode::kSBcnt1I32B32:
    case CompiledOpcode::kSBcnt1I32B64:
    case CompiledOpcode::kSFf0I32B32:
    case CompiledOpcode::kSFf0I32B64:
    case CompiledOpcode::kSFf1I32B32:
    case CompiledOpcode::kSFf1I32B64:
    case CompiledOpcode::kSFlbitI32B32:
    case CompiledOpcode::kSFlbitI32B64:
    case CompiledOpcode::kSFlbitI32:
    case CompiledOpcode::kSFlbitI32I64:
    case CompiledOpcode::kSBitreplicateB64B32:
    case CompiledOpcode::kSQuadmaskB32:
    case CompiledOpcode::kSQuadmaskB64:
    case CompiledOpcode::kSSextI32I8:
    case CompiledOpcode::kSSextI32I16:
    case CompiledOpcode::kSBitset0B32:
    case CompiledOpcode::kSBitset0B64:
    case CompiledOpcode::kSBitset1B32:
    case CompiledOpcode::kSBitset1B64:
      return ExecuteScalarMove(instruction, state, error_message);
    case CompiledOpcode::kSAndSaveexecB64:
    case CompiledOpcode::kSAndn1SaveexecB64:
    case CompiledOpcode::kSAndn2SaveexecB64:
    case CompiledOpcode::kSNandSaveexecB64:
    case CompiledOpcode::kSOrSaveexecB64:
    case CompiledOpcode::kSOrn1SaveexecB64:
    case CompiledOpcode::kSOrn2SaveexecB64:
    case CompiledOpcode::kSNorSaveexecB64:
    case CompiledOpcode::kSXorSaveexecB64:
    case CompiledOpcode::kSXnorSaveexecB64:
    case CompiledOpcode::kSAndn1WrexecB64:
    case CompiledOpcode::kSAndn2WrexecB64:
      return ExecuteExecMaskOp(instruction, state, error_message);
    case CompiledOpcode::kVMovB32:
    case CompiledOpcode::kVMovB64:
      return ExecuteVectorMove(instruction, state, error_message);
    case CompiledOpcode::kVReadfirstlaneB32:
    case CompiledOpcode::kVReadlaneB32:
      return ExecuteVectorToScalar(instruction, state, error_message);
    case CompiledOpcode::kVNotB32:
    case CompiledOpcode::kVBfrevB32:
    case CompiledOpcode::kVFfbhU32:
    case CompiledOpcode::kVFfblB32:
    case CompiledOpcode::kVFfbhI32:
    case CompiledOpcode::kVCvtF16U16:
    case CompiledOpcode::kVCvtF16I16:
    case CompiledOpcode::kVCvtU16F16:
    case CompiledOpcode::kVCvtI16F16:
    case CompiledOpcode::kVSatPkU8I16:
    case CompiledOpcode::kVCvtF32Ubyte0:
    case CompiledOpcode::kVCvtF32Ubyte1:
    case CompiledOpcode::kVCvtF32Ubyte2:
    case CompiledOpcode::kVCvtF32Ubyte3:
    case CompiledOpcode::kVCvtF32I32:
    case CompiledOpcode::kVCvtF32U32:
    case CompiledOpcode::kVCvtU32F32:
    case CompiledOpcode::kVCvtI32F32:
    case CompiledOpcode::kVCvtRpiI32F32:
    case CompiledOpcode::kVCvtFlrI32F32:
    case CompiledOpcode::kVCvtI32F64:
    case CompiledOpcode::kVCvtU32F64:
    case CompiledOpcode::kVCvtF16F32:
    case CompiledOpcode::kVCvtF32F16:
    case CompiledOpcode::kVCvtF32F64:
    case CompiledOpcode::kVCvtF64F32:
    case CompiledOpcode::kVCvtF64I32:
    case CompiledOpcode::kVCvtF64U32:
    case CompiledOpcode::kVRcpF16:
    case CompiledOpcode::kVSqrtF16:
    case CompiledOpcode::kVRsqF16:
    case CompiledOpcode::kVLogF16:
    case CompiledOpcode::kVExpF16:
    case CompiledOpcode::kVSinF16:
    case CompiledOpcode::kVCosF16:
    case CompiledOpcode::kVFrexpMantF16:
    case CompiledOpcode::kVFrexpExpI16F16:
    case CompiledOpcode::kVFractF16:
    case CompiledOpcode::kVTruncF16:
    case CompiledOpcode::kVCeilF16:
    case CompiledOpcode::kVRndneF16:
    case CompiledOpcode::kVFloorF16:
    case CompiledOpcode::kVExpF32:
    case CompiledOpcode::kVExpLegacyF32:
    case CompiledOpcode::kVLogF32:
    case CompiledOpcode::kVLogLegacyF32:
    case CompiledOpcode::kVRcpF32:
    case CompiledOpcode::kVRcpIflagF32:
    case CompiledOpcode::kVRsqF32:
    case CompiledOpcode::kVSqrtF32:
    case CompiledOpcode::kVSinF32:
    case CompiledOpcode::kVCosF32:
    case CompiledOpcode::kVFrexpExpI32F32:
    case CompiledOpcode::kVFrexpMantF32:
    case CompiledOpcode::kVFractF32:
    case CompiledOpcode::kVTruncF32:
    case CompiledOpcode::kVCeilF32:
    case CompiledOpcode::kVRndneF32:
    case CompiledOpcode::kVFloorF32:
    case CompiledOpcode::kVRcpF64:
    case CompiledOpcode::kVRsqF64:
    case CompiledOpcode::kVSqrtF64:
    case CompiledOpcode::kVFrexpExpI32F64:
    case CompiledOpcode::kVFrexpMantF64:
    case CompiledOpcode::kVFractF64:
    case CompiledOpcode::kVTruncF64:
    case CompiledOpcode::kVCeilF64:
    case CompiledOpcode::kVRndneF64:
    case CompiledOpcode::kVFloorF64:
      return ExecuteVectorUnary(instruction, state, error_message);
    case CompiledOpcode::kSAddU32:
    case CompiledOpcode::kSAddcU32:
    case CompiledOpcode::kSSubU32:
    case CompiledOpcode::kSSubbU32:
    case CompiledOpcode::kSMinI32:
    case CompiledOpcode::kSMinU32:
    case CompiledOpcode::kSMaxI32:
    case CompiledOpcode::kSMaxU32:
    case CompiledOpcode::kSMulI32:
    case CompiledOpcode::kSMulHiU32:
    case CompiledOpcode::kSMulHiI32:
    case CompiledOpcode::kSLshl1AddU32:
    case CompiledOpcode::kSLshl2AddU32:
    case CompiledOpcode::kSLshl3AddU32:
    case CompiledOpcode::kSLshl4AddU32:
    case CompiledOpcode::kSPackLlB32B16:
    case CompiledOpcode::kSPackLhB32B16:
    case CompiledOpcode::kSPackHhB32B16:
    case CompiledOpcode::kSCselectB32:
    case CompiledOpcode::kSCselectB64:
    case CompiledOpcode::kSAbsdiffI32:
    case CompiledOpcode::kSBfeU32:
    case CompiledOpcode::kSBfeI32:
    case CompiledOpcode::kSBfeU64:
    case CompiledOpcode::kSBfeI64:
    case CompiledOpcode::kSLshlB32:
    case CompiledOpcode::kSLshlB64:
    case CompiledOpcode::kSLshrB32:
    case CompiledOpcode::kSLshrB64:
    case CompiledOpcode::kSAshrI32:
    case CompiledOpcode::kSAshrI64:
    case CompiledOpcode::kSBfmB32:
    case CompiledOpcode::kSBfmB64:
    case CompiledOpcode::kSAndB32:
    case CompiledOpcode::kSAndn2B32:
    case CompiledOpcode::kSNandB32:
    case CompiledOpcode::kSAndB64:
    case CompiledOpcode::kSAndn2B64:
    case CompiledOpcode::kSNandB64:
    case CompiledOpcode::kSOrB32:
    case CompiledOpcode::kSOrn2B32:
    case CompiledOpcode::kSNorB32:
    case CompiledOpcode::kSOrB64:
    case CompiledOpcode::kSOrn2B64:
    case CompiledOpcode::kSNorB64:
    case CompiledOpcode::kSXorB32:
    case CompiledOpcode::kSXnorB32:
    case CompiledOpcode::kSXorB64:
    case CompiledOpcode::kSXnorB64:
      return ExecuteScalarBinary(instruction, state, error_message);
    case CompiledOpcode::kSCmpEqI32:
    case CompiledOpcode::kSCmpLgI32:
    case CompiledOpcode::kSCmpGtI32:
    case CompiledOpcode::kSCmpGeI32:
    case CompiledOpcode::kSCmpLtI32:
    case CompiledOpcode::kSCmpLeI32:
    case CompiledOpcode::kSCmpEqU32:
    case CompiledOpcode::kSCmpLgU32:
    case CompiledOpcode::kSCmpGtU32:
    case CompiledOpcode::kSCmpGeU32:
    case CompiledOpcode::kSCmpLtU32:
    case CompiledOpcode::kSCmpLeU32:
    case CompiledOpcode::kSCmpEqU64:
    case CompiledOpcode::kSCmpLgU64:
    case CompiledOpcode::kSBitcmp0B32:
    case CompiledOpcode::kSBitcmp1B32:
    case CompiledOpcode::kSBitcmp0B64:
    case CompiledOpcode::kSBitcmp1B64:
      return ExecuteScalarCompare(instruction, state, error_message);
    case CompiledOpcode::kSLoadDword:
    case CompiledOpcode::kSLoadDwordX2:
    case CompiledOpcode::kSStoreDword:
      return ExecuteScalarMemory(instruction, state, memory, error_message);
    case CompiledOpcode::kVCndmaskB32:
    case CompiledOpcode::kVMulLoU32:
    case CompiledOpcode::kVMulHiU32:
    case CompiledOpcode::kVMulHiI32:
    case CompiledOpcode::kVBcntU32B32:
    case CompiledOpcode::kVBfmB32:
    case CompiledOpcode::kVMbcntLoU32B32:
    case CompiledOpcode::kVMbcntHiU32B32:
    case CompiledOpcode::kVLshlRevB64:
    case CompiledOpcode::kVLshrRevB64:
    case CompiledOpcode::kVAshrRevI64:
    case CompiledOpcode::kVAddU32:
    case CompiledOpcode::kVAddCoU32:
    case CompiledOpcode::kVAddcCoU32:
    case CompiledOpcode::kVAddF16:
    case CompiledOpcode::kVAddF32:
    case CompiledOpcode::kVSubF16:
    case CompiledOpcode::kVSubF32:
    case CompiledOpcode::kVMulF16:
    case CompiledOpcode::kVMulF32:
    case CompiledOpcode::kVSubU32:
    case CompiledOpcode::kVSubCoU32:
    case CompiledOpcode::kVSubbCoU32:
    case CompiledOpcode::kVSubRevU32:
    case CompiledOpcode::kVSubRevCoU32:
    case CompiledOpcode::kVSubbrevCoU32:
    case CompiledOpcode::kVMinF16:
    case CompiledOpcode::kVMinF32:
    case CompiledOpcode::kVMaxF16:
    case CompiledOpcode::kVMaxF32:
    case CompiledOpcode::kVMinI32:
    case CompiledOpcode::kVMaxI32:
    case CompiledOpcode::kVMinU32:
    case CompiledOpcode::kVMaxU32:
    case CompiledOpcode::kVAddF64:
    case CompiledOpcode::kVMulF64:
    case CompiledOpcode::kVMinF64:
    case CompiledOpcode::kVMaxF64:
    case CompiledOpcode::kVLshrRevB32:
    case CompiledOpcode::kVAshrRevI32:
    case CompiledOpcode::kVLshlRevB32:
    case CompiledOpcode::kVWritelaneB32:
    case CompiledOpcode::kVAndB32:
    case CompiledOpcode::kVOrB32:
    case CompiledOpcode::kVXorB32:
      return ExecuteVectorBinary(instruction, state, error_message);
    case CompiledOpcode::kVAdd3U32:
    case CompiledOpcode::kVLshlAddU32:
    case CompiledOpcode::kVLshlAddU64:
    case CompiledOpcode::kVAddLshlU32:
    case CompiledOpcode::kVLshlOrB32:
    case CompiledOpcode::kVAndOrB32:
    case CompiledOpcode::kVOr3B32:
    case CompiledOpcode::kVXadU32:
    case CompiledOpcode::kVLerpU8:
    case CompiledOpcode::kVPermB32:
    case CompiledOpcode::kVBfeU32:
    case CompiledOpcode::kVBfeI32:
    case CompiledOpcode::kVBfiB32:
    case CompiledOpcode::kVAlignbitB32:
    case CompiledOpcode::kVAlignbyteB32:
    case CompiledOpcode::kVMin3I32:
    case CompiledOpcode::kVMin3U32:
    case CompiledOpcode::kVMax3I32:
    case CompiledOpcode::kVMax3U32:
    case CompiledOpcode::kVMed3I32:
    case CompiledOpcode::kVMed3U32:
    case CompiledOpcode::kVSadU8:
    case CompiledOpcode::kVSadHiU8:
    case CompiledOpcode::kVSadU16:
    case CompiledOpcode::kVSadU32:
    case CompiledOpcode::kVFmaF32:
    case CompiledOpcode::kVFmaF64:
    case CompiledOpcode::kVMadI32I24:
    case CompiledOpcode::kVMadU32U24:
    case CompiledOpcode::kVMadU64U32:
    case CompiledOpcode::kVMadI64I32:
      return ExecuteVectorTernary(instruction, state, error_message);
    case CompiledOpcode::kVCmpFF16:
    case CompiledOpcode::kVCmpLtF16:
    case CompiledOpcode::kVCmpEqF16:
    case CompiledOpcode::kVCmpLeF16:
    case CompiledOpcode::kVCmpGtF16:
    case CompiledOpcode::kVCmpLgF16:
    case CompiledOpcode::kVCmpGeF16:
    case CompiledOpcode::kVCmpOF16:
    case CompiledOpcode::kVCmpUF16:
    case CompiledOpcode::kVCmpNgeF16:
    case CompiledOpcode::kVCmpNlgF16:
    case CompiledOpcode::kVCmpNgtF16:
    case CompiledOpcode::kVCmpNleF16:
    case CompiledOpcode::kVCmpNeqF16:
    case CompiledOpcode::kVCmpNltF16:
    case CompiledOpcode::kVCmpTruF16:
    case CompiledOpcode::kVCmpClassF16:
    case CompiledOpcode::kVCmpxFF16:
    case CompiledOpcode::kVCmpxLtF16:
    case CompiledOpcode::kVCmpxEqF16:
    case CompiledOpcode::kVCmpxLeF16:
    case CompiledOpcode::kVCmpxGtF16:
    case CompiledOpcode::kVCmpxLgF16:
    case CompiledOpcode::kVCmpxGeF16:
    case CompiledOpcode::kVCmpxOF16:
    case CompiledOpcode::kVCmpxUF16:
    case CompiledOpcode::kVCmpxNgeF16:
    case CompiledOpcode::kVCmpxNlgF16:
    case CompiledOpcode::kVCmpxNgtF16:
    case CompiledOpcode::kVCmpxNleF16:
    case CompiledOpcode::kVCmpxNeqF16:
    case CompiledOpcode::kVCmpxNltF16:
    case CompiledOpcode::kVCmpxTruF16:
    case CompiledOpcode::kVCmpxClassF16:
    case CompiledOpcode::kVCmpFF32:
    case CompiledOpcode::kVCmpLtF32:
    case CompiledOpcode::kVCmpEqF32:
    case CompiledOpcode::kVCmpLeF32:
    case CompiledOpcode::kVCmpGtF32:
    case CompiledOpcode::kVCmpLgF32:
    case CompiledOpcode::kVCmpGeF32:
    case CompiledOpcode::kVCmpOF32:
    case CompiledOpcode::kVCmpUF32:
    case CompiledOpcode::kVCmpNgeF32:
    case CompiledOpcode::kVCmpNlgF32:
    case CompiledOpcode::kVCmpNgtF32:
    case CompiledOpcode::kVCmpNleF32:
    case CompiledOpcode::kVCmpNeqF32:
    case CompiledOpcode::kVCmpNltF32:
    case CompiledOpcode::kVCmpTruF32:
    case CompiledOpcode::kVCmpClassF32:
    case CompiledOpcode::kVCmpFF64:
    case CompiledOpcode::kVCmpLtF64:
    case CompiledOpcode::kVCmpEqF64:
    case CompiledOpcode::kVCmpLeF64:
    case CompiledOpcode::kVCmpGtF64:
    case CompiledOpcode::kVCmpLgF64:
    case CompiledOpcode::kVCmpGeF64:
    case CompiledOpcode::kVCmpOF64:
    case CompiledOpcode::kVCmpUF64:
    case CompiledOpcode::kVCmpNgeF64:
    case CompiledOpcode::kVCmpNlgF64:
    case CompiledOpcode::kVCmpNgtF64:
    case CompiledOpcode::kVCmpNleF64:
    case CompiledOpcode::kVCmpNeqF64:
    case CompiledOpcode::kVCmpNltF64:
    case CompiledOpcode::kVCmpTruF64:
    case CompiledOpcode::kVCmpClassF64:
    case CompiledOpcode::kVCmpxFF32:
    case CompiledOpcode::kVCmpxLtF32:
    case CompiledOpcode::kVCmpxEqF32:
    case CompiledOpcode::kVCmpxLeF32:
    case CompiledOpcode::kVCmpxGtF32:
    case CompiledOpcode::kVCmpxLgF32:
    case CompiledOpcode::kVCmpxGeF32:
    case CompiledOpcode::kVCmpxOF32:
    case CompiledOpcode::kVCmpxUF32:
    case CompiledOpcode::kVCmpxNgeF32:
    case CompiledOpcode::kVCmpxNlgF32:
    case CompiledOpcode::kVCmpxNgtF32:
    case CompiledOpcode::kVCmpxNleF32:
    case CompiledOpcode::kVCmpxNeqF32:
    case CompiledOpcode::kVCmpxNltF32:
    case CompiledOpcode::kVCmpxTruF32:
    case CompiledOpcode::kVCmpxClassF32:
    case CompiledOpcode::kVCmpxFF64:
    case CompiledOpcode::kVCmpxLtF64:
    case CompiledOpcode::kVCmpxEqF64:
    case CompiledOpcode::kVCmpxLeF64:
    case CompiledOpcode::kVCmpxGtF64:
    case CompiledOpcode::kVCmpxLgF64:
    case CompiledOpcode::kVCmpxGeF64:
    case CompiledOpcode::kVCmpxOF64:
    case CompiledOpcode::kVCmpxUF64:
    case CompiledOpcode::kVCmpxNgeF64:
    case CompiledOpcode::kVCmpxNlgF64:
    case CompiledOpcode::kVCmpxNgtF64:
    case CompiledOpcode::kVCmpxNleF64:
    case CompiledOpcode::kVCmpxNeqF64:
    case CompiledOpcode::kVCmpxNltF64:
    case CompiledOpcode::kVCmpxTruF64:
    case CompiledOpcode::kVCmpxClassF64:
    case CompiledOpcode::kVCmpEqI32:
    case CompiledOpcode::kVCmpNeI32:
    case CompiledOpcode::kVCmpLtI32:
    case CompiledOpcode::kVCmpLeI32:
    case CompiledOpcode::kVCmpGtI32:
    case CompiledOpcode::kVCmpGeI32:
    case CompiledOpcode::kVCmpEqU32:
    case CompiledOpcode::kVCmpNeU32:
    case CompiledOpcode::kVCmpLtU32:
    case CompiledOpcode::kVCmpLeU32:
    case CompiledOpcode::kVCmpGtU32:
    case CompiledOpcode::kVCmpGeU32:
    case CompiledOpcode::kVCmpFI64:
    case CompiledOpcode::kVCmpLtI64:
    case CompiledOpcode::kVCmpEqI64:
    case CompiledOpcode::kVCmpLeI64:
    case CompiledOpcode::kVCmpGtI64:
    case CompiledOpcode::kVCmpNeI64:
    case CompiledOpcode::kVCmpGeI64:
    case CompiledOpcode::kVCmpTI64:
    case CompiledOpcode::kVCmpFU64:
    case CompiledOpcode::kVCmpLtU64:
    case CompiledOpcode::kVCmpEqU64:
    case CompiledOpcode::kVCmpLeU64:
    case CompiledOpcode::kVCmpGtU64:
    case CompiledOpcode::kVCmpNeU64:
    case CompiledOpcode::kVCmpGeU64:
    case CompiledOpcode::kVCmpTU64:
    case CompiledOpcode::kVCmpxFI32:
    case CompiledOpcode::kVCmpxLtI32:
    case CompiledOpcode::kVCmpxEqI32:
    case CompiledOpcode::kVCmpxLeI32:
    case CompiledOpcode::kVCmpxGtI32:
    case CompiledOpcode::kVCmpxNeI32:
    case CompiledOpcode::kVCmpxGeI32:
    case CompiledOpcode::kVCmpxTI32:
    case CompiledOpcode::kVCmpxFU32:
    case CompiledOpcode::kVCmpxLtU32:
    case CompiledOpcode::kVCmpxEqU32:
    case CompiledOpcode::kVCmpxLeU32:
    case CompiledOpcode::kVCmpxGtU32:
    case CompiledOpcode::kVCmpxNeU32:
    case CompiledOpcode::kVCmpxGeU32:
    case CompiledOpcode::kVCmpxTU32:
    case CompiledOpcode::kVCmpxFI64:
    case CompiledOpcode::kVCmpxLtI64:
    case CompiledOpcode::kVCmpxEqI64:
    case CompiledOpcode::kVCmpxLeI64:
    case CompiledOpcode::kVCmpxGtI64:
    case CompiledOpcode::kVCmpxNeI64:
    case CompiledOpcode::kVCmpxGeI64:
    case CompiledOpcode::kVCmpxTI64:
    case CompiledOpcode::kVCmpxFU64:
    case CompiledOpcode::kVCmpxLtU64:
    case CompiledOpcode::kVCmpxEqU64:
    case CompiledOpcode::kVCmpxLeU64:
    case CompiledOpcode::kVCmpxGtU64:
    case CompiledOpcode::kVCmpxNeU64:
    case CompiledOpcode::kVCmpxGeU64:
    case CompiledOpcode::kVCmpxTU64:
      return ExecuteVectorCompare(instruction, state, error_message);
    case CompiledOpcode::kFlatLoadUByte:
    case CompiledOpcode::kFlatLoadSByte:
    case CompiledOpcode::kFlatLoadUShort:
    case CompiledOpcode::kFlatLoadSShort:
    case CompiledOpcode::kFlatLoadDword:
    case CompiledOpcode::kFlatLoadDwordX2:
    case CompiledOpcode::kFlatLoadDwordX3:
    case CompiledOpcode::kFlatLoadDwordX4:
    case CompiledOpcode::kFlatStoreByte:
    case CompiledOpcode::kFlatStoreShort:
    case CompiledOpcode::kFlatStoreDword:
    case CompiledOpcode::kFlatStoreDwordX2:
    case CompiledOpcode::kFlatStoreDwordX3:
    case CompiledOpcode::kFlatStoreDwordX4:
    case CompiledOpcode::kGlobalLoadUByte:
    case CompiledOpcode::kGlobalLoadSByte:
    case CompiledOpcode::kGlobalLoadUShort:
    case CompiledOpcode::kGlobalLoadSShort:
    case CompiledOpcode::kGlobalLoadDword:
    case CompiledOpcode::kGlobalLoadDwordX2:
    case CompiledOpcode::kGlobalLoadDwordX3:
    case CompiledOpcode::kGlobalLoadDwordX4:
    case CompiledOpcode::kGlobalStoreByte:
    case CompiledOpcode::kGlobalStoreShort:
    case CompiledOpcode::kGlobalStoreDword:
    case CompiledOpcode::kGlobalStoreDwordX2:
    case CompiledOpcode::kGlobalStoreDwordX3:
    case CompiledOpcode::kGlobalStoreDwordX4:
      return ExecuteVectorMemory(instruction, state, memory, error_message);
    case CompiledOpcode::kDsNop:
      return ValidateOperandCount(instruction, 0, error_message);
    case CompiledOpcode::kDsWriteB32:
    case CompiledOpcode::kDsReadB32:
    case CompiledOpcode::kDsAddU32:
    case CompiledOpcode::kDsSubU32:
    case CompiledOpcode::kDsRsubU32:
    case CompiledOpcode::kDsIncU32:
    case CompiledOpcode::kDsDecU32:
    case CompiledOpcode::kDsMinI32:
    case CompiledOpcode::kDsMaxI32:
    case CompiledOpcode::kDsMinU32:
    case CompiledOpcode::kDsMaxU32:
    case CompiledOpcode::kDsAndB32:
    case CompiledOpcode::kDsOrB32:
    case CompiledOpcode::kDsXorB32:
    case CompiledOpcode::kDsMskorB32:
    case CompiledOpcode::kDsCmpstB32:
    case CompiledOpcode::kDsCmpstF32:
    case CompiledOpcode::kDsAddF32:
    case CompiledOpcode::kDsMinF32:
    case CompiledOpcode::kDsMaxF32:
    case CompiledOpcode::kDsWriteB8:
    case CompiledOpcode::kDsWriteB16:
    case CompiledOpcode::kDsWrite2B32:
    case CompiledOpcode::kDsWrite2St64B32:
    case CompiledOpcode::kDsRead2B32:
    case CompiledOpcode::kDsRead2St64B32:
    case CompiledOpcode::kDsReadI8:
    case CompiledOpcode::kDsReadU8:
    case CompiledOpcode::kDsReadI16:
    case CompiledOpcode::kDsReadU16:
    case CompiledOpcode::kDsAddRtnU32:
    case CompiledOpcode::kDsSubRtnU32:
    case CompiledOpcode::kDsRsubRtnU32:
    case CompiledOpcode::kDsIncRtnU32:
    case CompiledOpcode::kDsDecRtnU32:
    case CompiledOpcode::kDsMinRtnI32:
    case CompiledOpcode::kDsMaxRtnI32:
    case CompiledOpcode::kDsMinRtnU32:
    case CompiledOpcode::kDsMaxRtnU32:
    case CompiledOpcode::kDsAndRtnB32:
    case CompiledOpcode::kDsOrRtnB32:
    case CompiledOpcode::kDsXorRtnB32:
    case CompiledOpcode::kDsMskorRtnB32:
    case CompiledOpcode::kDsWrxchgRtnB32:
    case CompiledOpcode::kDsCmpstRtnB32:
    case CompiledOpcode::kDsCmpstRtnF32:
    case CompiledOpcode::kDsWrapRtnB32:
    case CompiledOpcode::kDsAddRtnF32:
    case CompiledOpcode::kDsMinRtnF32:
    case CompiledOpcode::kDsMaxRtnF32:
      return ExecuteDsMemory(instruction, state, workgroup, error_message);
    case CompiledOpcode::kGlobalAtomicSwap:
    case CompiledOpcode::kGlobalAtomicCmpSwap:
    case CompiledOpcode::kGlobalAtomicAdd:
    case CompiledOpcode::kGlobalAtomicSub:
    case CompiledOpcode::kGlobalAtomicSMin:
    case CompiledOpcode::kGlobalAtomicUMin:
    case CompiledOpcode::kGlobalAtomicSMax:
    case CompiledOpcode::kGlobalAtomicUMax:
    case CompiledOpcode::kGlobalAtomicAnd:
    case CompiledOpcode::kGlobalAtomicOr:
    case CompiledOpcode::kGlobalAtomicXor:
    case CompiledOpcode::kGlobalAtomicInc:
    case CompiledOpcode::kGlobalAtomicDec:
    case CompiledOpcode::kGlobalAtomicAddF32:
    case CompiledOpcode::kGlobalAtomicPkAddF16:
    case CompiledOpcode::kGlobalAtomicAddF64:
    case CompiledOpcode::kGlobalAtomicMinF64:
    case CompiledOpcode::kGlobalAtomicMaxF64:
    case CompiledOpcode::kGlobalAtomicPkAddBf16:
    case CompiledOpcode::kGlobalAtomicSwapX2:
    case CompiledOpcode::kGlobalAtomicCmpSwapX2:
    case CompiledOpcode::kGlobalAtomicAddX2:
    case CompiledOpcode::kGlobalAtomicSubX2:
    case CompiledOpcode::kGlobalAtomicSMinX2:
    case CompiledOpcode::kGlobalAtomicUMinX2:
    case CompiledOpcode::kGlobalAtomicSMaxX2:
    case CompiledOpcode::kGlobalAtomicUMaxX2:
    case CompiledOpcode::kGlobalAtomicAndX2:
    case CompiledOpcode::kGlobalAtomicOrX2:
    case CompiledOpcode::kGlobalAtomicXorX2:
    case CompiledOpcode::kGlobalAtomicIncX2:
    case CompiledOpcode::kGlobalAtomicDecX2:
      return ExecuteGlobalAtomic(instruction, state, memory, error_message);
    case CompiledOpcode::kSBranch:
    case CompiledOpcode::kSCbranchScc0:
    case CompiledOpcode::kSCbranchScc1:
    case CompiledOpcode::kSCbranchVccz:
    case CompiledOpcode::kSCbranchVccnz:
    case CompiledOpcode::kSCbranchExecz:
    case CompiledOpcode::kSCbranchExecnz:
      return ExecuteBranch(instruction, state, pc_was_updated, error_message);
  }

  if (error_message != nullptr) {
    *error_message = "unsupported compiled opcode";
  }
  return false;
}

bool Gfx950Interpreter::ExecuteScalarMove(const DecodedInstruction& instruction,
                                          WaveExecutionState* state,
                                          std::string* error_message) const {
  if (!ValidateOperandCount(instruction, 2, error_message)) {
    return false;
  }

  if (instruction.opcode == "S_CMOV_B32" ||
      instruction.opcode == "S_CMOVK_I32") {
    if (!state->scc) {
      if (error_message != nullptr) {
        error_message->clear();
      }
      return true;
    }
    const std::uint32_t value =
        ReadScalarOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    return WriteScalarOperand(instruction.operands[0], value, state, error_message);
  }

  if (instruction.opcode == "S_NOT_B32" ||
      instruction.opcode == "S_ABS_I32" ||
      instruction.opcode == "S_BREV_B32" ||
      instruction.opcode == "S_BCNT0_I32_B32" ||
      instruction.opcode == "S_BCNT1_I32_B32" ||
      instruction.opcode == "S_FF0_I32_B32" ||
      instruction.opcode == "S_FF1_I32_B32" ||
      instruction.opcode == "S_FLBIT_I32_B32" ||
      instruction.opcode == "S_FLBIT_I32" ||
      instruction.opcode == "S_QUADMASK_B32" ||
      instruction.opcode == "S_SEXT_I32_I8" ||
      instruction.opcode == "S_SEXT_I32_I16") {
    std::uint32_t value =
        ReadScalarOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    bool update_scc = false;
    if (instruction.opcode == "S_NOT_B32") {
      value = ~value;
      update_scc = true;
    } else if (instruction.opcode == "S_ABS_I32") {
      const std::int32_t signed_value = BitCast<std::int32_t>(value);
      const std::int64_t abs_value =
          signed_value < 0 ? -static_cast<std::int64_t>(signed_value)
                           : static_cast<std::int64_t>(signed_value);
      value = static_cast<std::uint32_t>(abs_value);
      update_scc = true;
    } else if (instruction.opcode == "S_BREV_B32") {
      value = ReverseBits32(value);
    } else if (instruction.opcode == "S_BCNT0_I32_B32") {
      value = 32u - PopCount32(value);
      update_scc = true;
    } else if (instruction.opcode == "S_BCNT1_I32_B32") {
      value = PopCount32(value);
      update_scc = true;
    } else if (instruction.opcode == "S_FF0_I32_B32") {
      value = FindFirstBitLow(~value);
    } else if (instruction.opcode == "S_FF1_I32_B32") {
      value = FindFirstBitLow(value);
    } else if (instruction.opcode == "S_FLBIT_I32_B32") {
      value = FindFirstBitHighUnsigned(value);
    } else if (instruction.opcode == "S_FLBIT_I32") {
      value = FindFirstBitHighSigned(value);
    } else if (instruction.opcode == "S_QUADMASK_B32") {
      value = static_cast<std::uint32_t>(ReduceQuadMask(value, 32u));
      update_scc = true;
    } else if (instruction.opcode == "S_SEXT_I32_I8") {
      value = static_cast<std::uint32_t>(
          static_cast<std::int32_t>(static_cast<std::int8_t>(value & 0xffu)));
    } else {
      value = static_cast<std::uint32_t>(
          static_cast<std::int32_t>(static_cast<std::int16_t>(value & 0xffffu)));
    }
    if (update_scc) {
      state->scc = value != 0;
    }
    return WriteScalarOperand(instruction.operands[0], value, state, error_message);
  }

  if (instruction.opcode == "S_BITSET0_B32" ||
      instruction.opcode == "S_BITSET1_B32") {
    const std::uint32_t bit_offset =
        ReadScalarOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint32_t current_value =
        ReadScalarOperand(instruction.operands[0], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint32_t bit_mask = 1u << (bit_offset & 31u);
    const std::uint32_t value =
        instruction.opcode == "S_BITSET0_B32" ? (current_value & ~bit_mask)
                                              : (current_value | bit_mask);
    return WriteScalarOperand(instruction.operands[0], value, state, error_message);
  }

  if (instruction.opcode == "S_BITSET0_B64" ||
      instruction.opcode == "S_BITSET1_B64") {
    const std::uint32_t bit_offset =
        ReadScalarOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint64_t current_value =
        ReadScalarPairOperand(instruction.operands[0], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint64_t bit_mask = std::uint64_t{1} << (bit_offset & 63u);
    const std::uint64_t value =
        instruction.opcode == "S_BITSET0_B64" ? (current_value & ~bit_mask)
                                              : (current_value | bit_mask);
    return WriteScalarPairOperand(instruction.operands[0], value, state,
                                  error_message);
  }

  if (instruction.opcode == "S_BCNT0_I32_B64" ||
      instruction.opcode == "S_BCNT1_I32_B64" ||
      instruction.opcode == "S_FF0_I32_B64" ||
      instruction.opcode == "S_FF1_I32_B64" ||
      instruction.opcode == "S_FLBIT_I32_B64" ||
      instruction.opcode == "S_FLBIT_I32_I64") {
    const std::uint64_t source =
        ReadScalarPairOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    std::uint32_t value = 0;
    bool update_scc = false;
    if (instruction.opcode == "S_BCNT0_I32_B64") {
      value = 64u - PopCount64(source);
      update_scc = true;
    } else if (instruction.opcode == "S_BCNT1_I32_B64") {
      value = PopCount64(source);
      update_scc = true;
    } else if (instruction.opcode == "S_FF0_I32_B64") {
      value = FindFirstBitLow64(~source);
    } else if (instruction.opcode == "S_FF1_I32_B64") {
      value = FindFirstBitLow64(source);
    } else if (instruction.opcode == "S_FLBIT_I32_B64") {
      value = FindFirstBitHighUnsigned64(source);
    } else {
      value = FindFirstBitHighSigned64(source);
    }
    if (update_scc) {
      state->scc = value != 0;
    }
    return WriteScalarOperand(instruction.operands[0], value, state, error_message);
  }

  if (instruction.opcode == "S_CMOV_B64") {
    if (!state->scc) {
      if (error_message != nullptr) {
        error_message->clear();
      }
      return true;
    }
    const std::uint64_t value =
        ReadScalarPairOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    return WriteScalarPairOperand(instruction.operands[0], value, state,
                                  error_message);
  }

  if (instruction.opcode == "S_BITREPLICATE_B64_B32") {
    const std::uint32_t value =
        ReadScalarOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    return WriteScalarPairOperand(instruction.operands[0],
                                  BitReplicate32To64(value), state,
                                  error_message);
  }

  if (instruction.opcode == "S_QUADMASK_B64") {
    const std::uint64_t source =
        ReadScalarPairOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint64_t value = ReduceQuadMask(source, 64u);
    state->scc = value != 0;
    return WriteScalarPairOperand(instruction.operands[0], value, state,
                                  error_message);
  }

  if (instruction.opcode == "S_MOV_B64" || instruction.opcode == "S_NOT_B64" ||
      instruction.opcode == "S_BREV_B64") {
    std::uint64_t value =
        ReadScalarPairOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    if (instruction.opcode == "S_NOT_B64") {
      value = ~value;
      state->scc = (value != 0);
    } else if (instruction.opcode == "S_BREV_B64") {
      value = ReverseBits64(value);
    } else {
      state->scc = (value != 0);
    }
    return WriteScalarPairOperand(instruction.operands[0], value, state,
                                  error_message);
  }

  const std::uint32_t value =
      ReadScalarOperand(instruction.operands[1], *state, error_message);
  if (error_message != nullptr && !error_message->empty()) {
    return false;
  }
  state->scc = (value != 0);
  return WriteScalarOperand(instruction.operands[0], value, state, error_message);
}

bool Gfx950Interpreter::ExecuteScalarMove(const CompiledInstruction& instruction,
                                          WaveExecutionState* state,
                                          std::string* error_message) const {
  if (!ValidateOperandCount(instruction, 2, error_message)) {
    return false;
  }

  if (instruction.opcode == CompiledOpcode::kSCmovB32) {
    if (!state->scc) {
      if (error_message != nullptr) {
        error_message->clear();
      }
      return true;
    }
    const std::uint32_t value =
        ReadScalarOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    return WriteScalarOperand(instruction.operands[0], value, state, error_message);
  }

  if (instruction.opcode == CompiledOpcode::kSNotB32 ||
      instruction.opcode == CompiledOpcode::kSAbsI32 ||
      instruction.opcode == CompiledOpcode::kSBrevB32 ||
      instruction.opcode == CompiledOpcode::kSBcnt0I32B32 ||
      instruction.opcode == CompiledOpcode::kSBcnt1I32B32 ||
      instruction.opcode == CompiledOpcode::kSFf0I32B32 ||
      instruction.opcode == CompiledOpcode::kSFf1I32B32 ||
      instruction.opcode == CompiledOpcode::kSFlbitI32B32 ||
      instruction.opcode == CompiledOpcode::kSFlbitI32 ||
      instruction.opcode == CompiledOpcode::kSQuadmaskB32 ||
      instruction.opcode == CompiledOpcode::kSSextI32I8 ||
      instruction.opcode == CompiledOpcode::kSSextI32I16) {
    std::uint32_t value =
        ReadScalarOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    bool update_scc = false;
    if (instruction.opcode == CompiledOpcode::kSNotB32) {
      value = ~value;
      update_scc = true;
    } else if (instruction.opcode == CompiledOpcode::kSAbsI32) {
      const std::int32_t signed_value = BitCast<std::int32_t>(value);
      const std::int64_t abs_value =
          signed_value < 0 ? -static_cast<std::int64_t>(signed_value)
                           : static_cast<std::int64_t>(signed_value);
      value = static_cast<std::uint32_t>(abs_value);
      update_scc = true;
    } else if (instruction.opcode == CompiledOpcode::kSBrevB32) {
      value = ReverseBits32(value);
    } else if (instruction.opcode == CompiledOpcode::kSBcnt0I32B32) {
      value = 32u - PopCount32(value);
      update_scc = true;
    } else if (instruction.opcode == CompiledOpcode::kSBcnt1I32B32) {
      value = PopCount32(value);
      update_scc = true;
    } else if (instruction.opcode == CompiledOpcode::kSFf0I32B32) {
      value = FindFirstBitLow(~value);
    } else if (instruction.opcode == CompiledOpcode::kSFf1I32B32) {
      value = FindFirstBitLow(value);
    } else if (instruction.opcode == CompiledOpcode::kSFlbitI32B32) {
      value = FindFirstBitHighUnsigned(value);
    } else if (instruction.opcode == CompiledOpcode::kSFlbitI32) {
      value = FindFirstBitHighSigned(value);
    } else if (instruction.opcode == CompiledOpcode::kSQuadmaskB32) {
      value = static_cast<std::uint32_t>(ReduceQuadMask(value, 32u));
      update_scc = true;
    } else if (instruction.opcode == CompiledOpcode::kSSextI32I8) {
      value = static_cast<std::uint32_t>(
          static_cast<std::int32_t>(static_cast<std::int8_t>(value & 0xffu)));
    } else {
      value = static_cast<std::uint32_t>(
          static_cast<std::int32_t>(static_cast<std::int16_t>(value & 0xffffu)));
    }
    if (update_scc) {
      state->scc = value != 0;
    }
    return WriteScalarOperand(instruction.operands[0], value, state, error_message);
  }

  if (instruction.opcode == CompiledOpcode::kSBitset0B32 ||
      instruction.opcode == CompiledOpcode::kSBitset1B32) {
    const std::uint32_t bit_offset =
        ReadScalarOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint32_t current_value =
        ReadScalarOperand(instruction.operands[0], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint32_t bit_mask = 1u << (bit_offset & 31u);
    const std::uint32_t value =
        instruction.opcode == CompiledOpcode::kSBitset0B32
            ? (current_value & ~bit_mask)
            : (current_value | bit_mask);
    return WriteScalarOperand(instruction.operands[0], value, state, error_message);
  }

  if (instruction.opcode == CompiledOpcode::kSBitset0B64 ||
      instruction.opcode == CompiledOpcode::kSBitset1B64) {
    const std::uint32_t bit_offset =
        ReadScalarOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint64_t current_value =
        ReadScalarPairOperand(instruction.operands[0], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint64_t bit_mask = std::uint64_t{1} << (bit_offset & 63u);
    const std::uint64_t value =
        instruction.opcode == CompiledOpcode::kSBitset0B64
            ? (current_value & ~bit_mask)
            : (current_value | bit_mask);
    return WriteScalarPairOperand(instruction.operands[0], value, state,
                                  error_message);
  }

  if (instruction.opcode == CompiledOpcode::kSBcnt0I32B64 ||
      instruction.opcode == CompiledOpcode::kSBcnt1I32B64 ||
      instruction.opcode == CompiledOpcode::kSFf0I32B64 ||
      instruction.opcode == CompiledOpcode::kSFf1I32B64 ||
      instruction.opcode == CompiledOpcode::kSFlbitI32B64 ||
      instruction.opcode == CompiledOpcode::kSFlbitI32I64) {
    const std::uint64_t source =
        ReadScalarPairOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    std::uint32_t value = 0;
    bool update_scc = false;
    if (instruction.opcode == CompiledOpcode::kSBcnt0I32B64) {
      value = 64u - PopCount64(source);
      update_scc = true;
    } else if (instruction.opcode == CompiledOpcode::kSBcnt1I32B64) {
      value = PopCount64(source);
      update_scc = true;
    } else if (instruction.opcode == CompiledOpcode::kSFf0I32B64) {
      value = FindFirstBitLow64(~source);
    } else if (instruction.opcode == CompiledOpcode::kSFf1I32B64) {
      value = FindFirstBitLow64(source);
    } else if (instruction.opcode == CompiledOpcode::kSFlbitI32B64) {
      value = FindFirstBitHighUnsigned64(source);
    } else {
      value = FindFirstBitHighSigned64(source);
    }
    if (update_scc) {
      state->scc = value != 0;
    }
    return WriteScalarOperand(instruction.operands[0], value, state, error_message);
  }

  if (instruction.opcode == CompiledOpcode::kSCmovB64) {
    if (!state->scc) {
      if (error_message != nullptr) {
        error_message->clear();
      }
      return true;
    }
    const std::uint64_t value =
        ReadScalarPairOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    return WriteScalarPairOperand(instruction.operands[0], value, state,
                                  error_message);
  }

  if (instruction.opcode == CompiledOpcode::kSBitreplicateB64B32) {
    const std::uint32_t value =
        ReadScalarOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    return WriteScalarPairOperand(instruction.operands[0],
                                  BitReplicate32To64(value), state,
                                  error_message);
  }

  if (instruction.opcode == CompiledOpcode::kSQuadmaskB64) {
    const std::uint64_t source =
        ReadScalarPairOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint64_t value = ReduceQuadMask(source, 64u);
    state->scc = value != 0;
    return WriteScalarPairOperand(instruction.operands[0], value, state,
                                  error_message);
  }

  if (instruction.opcode == CompiledOpcode::kSMovB64 ||
      instruction.opcode == CompiledOpcode::kSNotB64 ||
      instruction.opcode == CompiledOpcode::kSBrevB64) {
    std::uint64_t value =
        ReadScalarPairOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    if (instruction.opcode == CompiledOpcode::kSNotB64) {
      value = ~value;
      state->scc = (value != 0);
    } else if (instruction.opcode == CompiledOpcode::kSBrevB64) {
      value = ReverseBits64(value);
    } else {
      state->scc = (value != 0);
    }
    return WriteScalarPairOperand(instruction.operands[0], value, state,
                                  error_message);
  }

  const std::uint32_t value =
      ReadScalarOperand(instruction.operands[1], *state, error_message);
  if (error_message != nullptr && !error_message->empty()) {
    return false;
  }
  state->scc = (value != 0);
  return WriteScalarOperand(instruction.operands[0], value, state, error_message);
}

bool Gfx950Interpreter::ExecuteExecMaskOp(const DecodedInstruction& instruction,
                                          WaveExecutionState* state,
                                          std::string* error_message) const {
  if (!ValidateOperandCount(instruction, 2, error_message)) {
    return false;
  }

  const std::uint64_t source =
      ReadScalarPairOperand(instruction.operands[1], *state, error_message);
  if (error_message != nullptr && !error_message->empty()) {
    return false;
  }

  const std::uint64_t previous_exec = state->exec_mask;
  std::uint64_t next_exec = previous_exec;
  if (!ApplyExecMaskOpcode(instruction.opcode, previous_exec, source,
                           &next_exec)) {
    if (error_message != nullptr) {
      *error_message = "unsupported exec mask opcode";
    }
    return false;
  }

  const std::uint64_t destination_value =
      IsWrExecOpcode(instruction.opcode) ? next_exec : previous_exec;
  if (!WriteScalarPairOperand(instruction.operands[0], destination_value, state,
                              error_message)) {
    return false;
  }
  state->exec_mask = next_exec;
  state->scc = next_exec != 0;
  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool Gfx950Interpreter::ExecuteExecMaskOp(const CompiledInstruction& instruction,
                                          WaveExecutionState* state,
                                          std::string* error_message) const {
  if (!ValidateOperandCount(instruction, 2, error_message)) {
    return false;
  }

  const std::uint64_t source =
      ReadScalarPairOperand(instruction.operands[1], *state, error_message);
  if (error_message != nullptr && !error_message->empty()) {
    return false;
  }

  const std::uint64_t previous_exec = state->exec_mask;
  std::uint64_t next_exec = previous_exec;
  if (!ApplyExecMaskOpcode(instruction.opcode, previous_exec, source,
                           &next_exec)) {
    if (error_message != nullptr) {
      *error_message = "unsupported compiled exec mask opcode";
    }
    return false;
  }

  const std::uint64_t destination_value =
      IsWrExecOpcode(instruction.opcode) ? next_exec : previous_exec;
  if (!WriteScalarPairOperand(instruction.operands[0], destination_value, state,
                              error_message)) {
    return false;
  }
  state->exec_mask = next_exec;
  state->scc = next_exec != 0;
  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool Gfx950Interpreter::ExecuteScalarBinary(const DecodedInstruction& instruction,
                                            WaveExecutionState* state,
                                            std::string* error_message) const {
  if (!ValidateOperandCount(instruction, 3, error_message)) {
    return false;
  }

  if (instruction.opcode == "S_BFE_U64" || instruction.opcode == "S_BFE_I64") {
    const std::uint64_t lhs =
        ReadScalarPairOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint32_t rhs =
        ReadScalarOperand(instruction.operands[2], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint64_t result =
        instruction.opcode == "S_BFE_U64" ? ExtractUnsignedBitfield64(lhs, rhs)
                                          : ExtractSignedBitfield64(lhs, rhs);
    state->scc = result != 0;
    return WriteScalarPairOperand(instruction.operands[0], result, state,
                                  error_message);
  }

  if (instruction.opcode == "S_CSELECT_B64") {
    const std::uint64_t lhs =
        ReadScalarPairOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint64_t rhs =
        ReadScalarPairOperand(instruction.operands[2], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint64_t result = state->scc ? lhs : rhs;
    return WriteScalarPairOperand(instruction.operands[0], result, state,
                                  error_message);
  }

  if (instruction.opcode == "S_LSHL_B64" || instruction.opcode == "S_LSHR_B64" ||
      instruction.opcode == "S_ASHR_I64" || instruction.opcode == "S_BFM_B64") {
    const std::uint64_t lhs =
        instruction.opcode == "S_BFM_B64"
            ? 0
            : ReadScalarPairOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint32_t rhs =
        ReadScalarOperand(instruction.operands[2], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }

    std::uint64_t result = 0;
    bool update_scc = true;
    if (instruction.opcode == "S_LSHL_B64") {
      result = lhs << (rhs & 63u);
    } else if (instruction.opcode == "S_LSHR_B64") {
      result = lhs >> (rhs & 63u);
    } else if (instruction.opcode == "S_ASHR_I64") {
      result = static_cast<std::uint64_t>(
          BitCast<std::int64_t>(lhs) >> (rhs & 63u));
    } else {
      const std::uint32_t width =
          ReadScalarOperand(instruction.operands[1], *state, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      result = MakeBitfieldMask64(width, rhs);
      update_scc = false;
    }

    if (update_scc) {
      state->scc = result != 0;
    }
    return WriteScalarPairOperand(instruction.operands[0], result, state,
                                  error_message);
  }

  if (instruction.opcode == "S_AND_B64" || instruction.opcode == "S_ANDN2_B64" ||
      instruction.opcode == "S_NAND_B64" || instruction.opcode == "S_OR_B64" ||
      instruction.opcode == "S_ORN2_B64" || instruction.opcode == "S_NOR_B64" ||
      instruction.opcode == "S_XOR_B64" || instruction.opcode == "S_XNOR_B64") {
    const std::uint64_t lhs =
        ReadScalarPairOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint64_t rhs =
        ReadScalarPairOperand(instruction.operands[2], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }

    std::uint64_t result = 0;
    if (instruction.opcode == "S_AND_B64") {
      result = lhs & rhs;
    } else if (instruction.opcode == "S_ANDN2_B64") {
      result = lhs & ~rhs;
    } else if (instruction.opcode == "S_NAND_B64") {
      result = ~(lhs & rhs);
    } else if (instruction.opcode == "S_OR_B64") {
      result = lhs | rhs;
    } else if (instruction.opcode == "S_ORN2_B64") {
      result = lhs | ~rhs;
    } else if (instruction.opcode == "S_NOR_B64") {
      result = ~(lhs | rhs);
    } else if (instruction.opcode == "S_XOR_B64") {
      result = lhs ^ rhs;
    } else {
      result = ~(lhs ^ rhs);
    }

    state->scc = result != 0;
    return WriteScalarPairOperand(instruction.operands[0], result, state,
                                  error_message);
  }

  const std::uint32_t lhs =
      ReadScalarOperand(instruction.operands[1], *state, error_message);
  if (error_message != nullptr && !error_message->empty()) {
    return false;
  }
  const std::uint32_t rhs =
      ReadScalarOperand(instruction.operands[2], *state, error_message);
  if (error_message != nullptr && !error_message->empty()) {
    return false;
  }

  std::uint32_t result = 0;
  bool scc = state->scc;
  bool update_scc = true;
  if (instruction.opcode == "S_ADD_U32" || instruction.opcode == "S_ADD_I32" ||
      instruction.opcode == "S_ADDK_I32") {
    const std::uint64_t wide = static_cast<std::uint64_t>(lhs) + rhs;
    result = static_cast<std::uint32_t>(wide);
    scc = wide > std::numeric_limits<std::uint32_t>::max();
  } else if (instruction.opcode == "S_ADDC_U32") {
    const std::uint64_t wide =
        static_cast<std::uint64_t>(lhs) + rhs + (state->scc ? 1u : 0u);
    result = static_cast<std::uint32_t>(wide);
    scc = wide > std::numeric_limits<std::uint32_t>::max();
  } else if (instruction.opcode == "S_SUB_U32" ||
             instruction.opcode == "S_SUB_I32") {
    result = lhs - rhs;
    scc = lhs >= rhs;
  } else if (instruction.opcode == "S_SUBB_U32") {
    const std::uint64_t subtrahend =
        static_cast<std::uint64_t>(rhs) + (state->scc ? 1u : 0u);
    result = static_cast<std::uint32_t>(static_cast<std::uint64_t>(lhs) - subtrahend);
    scc = static_cast<std::uint64_t>(lhs) >= subtrahend;
  } else if (instruction.opcode == "S_MUL_I32" ||
             instruction.opcode == "S_MULK_I32") {
    const std::int64_t product =
        static_cast<std::int64_t>(BitCast<std::int32_t>(lhs)) *
        static_cast<std::int64_t>(BitCast<std::int32_t>(rhs));
    result = static_cast<std::uint32_t>(product);
    update_scc = false;
  } else if (instruction.opcode == "S_MUL_HI_U32") {
    result = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(lhs) * static_cast<std::uint64_t>(rhs)) >> 32);
    update_scc = false;
  } else if (instruction.opcode == "S_MUL_HI_I32") {
    const std::int64_t product =
        static_cast<std::int64_t>(BitCast<std::int32_t>(lhs)) *
        static_cast<std::int64_t>(BitCast<std::int32_t>(rhs));
    result = static_cast<std::uint32_t>(static_cast<std::uint64_t>(product) >> 32);
    update_scc = false;
  } else if (instruction.opcode == "S_LSHL1_ADD_U32" ||
             instruction.opcode == "S_LSHL2_ADD_U32" ||
             instruction.opcode == "S_LSHL3_ADD_U32" ||
             instruction.opcode == "S_LSHL4_ADD_U32") {
    const std::uint32_t shift_amount =
        instruction.opcode == "S_LSHL1_ADD_U32"
            ? 1u
            : (instruction.opcode == "S_LSHL2_ADD_U32"
                   ? 2u
                   : (instruction.opcode == "S_LSHL3_ADD_U32" ? 3u : 4u));
    const std::uint32_t shifted = lhs << shift_amount;
    const std::uint64_t wide = static_cast<std::uint64_t>(shifted) + rhs;
    result = static_cast<std::uint32_t>(wide);
    scc = wide > std::numeric_limits<std::uint32_t>::max();
  } else if (instruction.opcode == "S_PACK_LL_B32_B16") {
    result = (lhs & 0xffffu) | ((rhs & 0xffffu) << 16);
    update_scc = false;
  } else if (instruction.opcode == "S_PACK_LH_B32_B16") {
    result = (lhs & 0xffffu) | (rhs & 0xffff0000u);
    update_scc = false;
  } else if (instruction.opcode == "S_PACK_HH_B32_B16") {
    result = ((lhs >> 16) & 0xffffu) | (rhs & 0xffff0000u);
    update_scc = false;
  } else if (instruction.opcode == "S_MIN_I32") {
    const std::int32_t lhs_signed = BitCast<std::int32_t>(lhs);
    const std::int32_t rhs_signed = BitCast<std::int32_t>(rhs);
    const bool select_lhs = lhs_signed <= rhs_signed;
    result = select_lhs ? lhs : rhs;
    scc = select_lhs;
  } else if (instruction.opcode == "S_MIN_U32") {
    const bool select_lhs = lhs <= rhs;
    result = select_lhs ? lhs : rhs;
    scc = select_lhs;
  } else if (instruction.opcode == "S_MAX_I32") {
    const std::int32_t lhs_signed = BitCast<std::int32_t>(lhs);
    const std::int32_t rhs_signed = BitCast<std::int32_t>(rhs);
    const bool select_lhs = lhs_signed >= rhs_signed;
    result = select_lhs ? lhs : rhs;
    scc = select_lhs;
  } else if (instruction.opcode == "S_MAX_U32") {
    const bool select_lhs = lhs >= rhs;
    result = select_lhs ? lhs : rhs;
    scc = select_lhs;
  } else if (instruction.opcode == "S_CSELECT_B32") {
    result = state->scc ? lhs : rhs;
    update_scc = false;
  } else if (instruction.opcode == "S_ABSDIFF_I32") {
    const std::int64_t diff =
        static_cast<std::int64_t>(BitCast<std::int32_t>(lhs)) -
        static_cast<std::int64_t>(BitCast<std::int32_t>(rhs));
    result = static_cast<std::uint32_t>(diff < 0 ? -diff : diff);
    scc = result != 0;
  } else if (instruction.opcode == "S_BFE_U32") {
    result = ExtractUnsignedBitfield32(lhs, rhs);
    scc = result != 0;
  } else if (instruction.opcode == "S_BFE_I32") {
    result = ExtractSignedBitfield32(lhs, rhs);
    scc = result != 0;
  } else if (instruction.opcode == "S_LSHL_B32") {
    result = lhs << (rhs & 31u);
    scc = result != 0;
  } else if (instruction.opcode == "S_LSHR_B32") {
    result = lhs >> (rhs & 31u);
    scc = result != 0;
  } else if (instruction.opcode == "S_ASHR_I32") {
    result = static_cast<std::uint32_t>(
        BitCast<std::int32_t>(lhs) >> (rhs & 31u));
    scc = result != 0;
  } else if (instruction.opcode == "S_BFM_B32") {
    result = MakeBitfieldMask32(lhs, rhs);
    update_scc = false;
  } else if (instruction.opcode == "S_AND_B32") {
    result = lhs & rhs;
    scc = result != 0;
  } else if (instruction.opcode == "S_ANDN2_B32") {
    result = lhs & ~rhs;
    scc = result != 0;
  } else if (instruction.opcode == "S_NAND_B32") {
    result = ~(lhs & rhs);
    scc = result != 0;
  } else if (instruction.opcode == "S_OR_B32") {
    result = lhs | rhs;
    scc = result != 0;
  } else if (instruction.opcode == "S_ORN2_B32") {
    result = lhs | ~rhs;
    scc = result != 0;
  } else if (instruction.opcode == "S_NOR_B32") {
    result = ~(lhs | rhs);
    scc = result != 0;
  } else if (instruction.opcode == "S_XOR_B32") {
    result = lhs ^ rhs;
    scc = result != 0;
  } else if (instruction.opcode == "S_XNOR_B32") {
    result = ~(lhs ^ rhs);
    scc = result != 0;
  } else {
    if (error_message != nullptr) {
      *error_message = "unsupported scalar binary opcode";
    }
    return false;
  }

  if (update_scc) {
    state->scc = scc;
  }
  return WriteScalarOperand(instruction.operands[0], result, state, error_message);
}

bool Gfx950Interpreter::ExecuteScalarBinary(const CompiledInstruction& instruction,
                                            WaveExecutionState* state,
                                            std::string* error_message) const {
  if (!ValidateOperandCount(instruction, 3, error_message)) {
    return false;
  }

  if (instruction.opcode == CompiledOpcode::kSBfeU64 ||
      instruction.opcode == CompiledOpcode::kSBfeI64) {
    const std::uint64_t lhs =
        ReadScalarPairOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint32_t rhs =
        ReadScalarOperand(instruction.operands[2], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint64_t result =
        instruction.opcode == CompiledOpcode::kSBfeU64
            ? ExtractUnsignedBitfield64(lhs, rhs)
            : ExtractSignedBitfield64(lhs, rhs);
    state->scc = result != 0;
    return WriteScalarPairOperand(instruction.operands[0], result, state,
                                  error_message);
  }

  if (instruction.opcode == CompiledOpcode::kSCselectB64) {
    const std::uint64_t lhs =
        ReadScalarPairOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint64_t rhs =
        ReadScalarPairOperand(instruction.operands[2], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint64_t result = state->scc ? lhs : rhs;
    return WriteScalarPairOperand(instruction.operands[0], result, state,
                                  error_message);
  }

  if (instruction.opcode == CompiledOpcode::kSLshlB64 ||
      instruction.opcode == CompiledOpcode::kSLshrB64 ||
      instruction.opcode == CompiledOpcode::kSAshrI64 ||
      instruction.opcode == CompiledOpcode::kSBfmB64) {
    const std::uint64_t lhs =
        instruction.opcode == CompiledOpcode::kSBfmB64
            ? 0
            : ReadScalarPairOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint32_t rhs =
        ReadScalarOperand(instruction.operands[2], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }

    std::uint64_t result = 0;
    bool update_scc = true;
    switch (instruction.opcode) {
      case CompiledOpcode::kSLshlB64:
        result = lhs << (rhs & 63u);
        break;
      case CompiledOpcode::kSLshrB64:
        result = lhs >> (rhs & 63u);
        break;
      case CompiledOpcode::kSAshrI64:
        result = static_cast<std::uint64_t>(
            BitCast<std::int64_t>(lhs) >> (rhs & 63u));
        break;
      case CompiledOpcode::kSBfmB64: {
        const std::uint32_t width =
            ReadScalarOperand(instruction.operands[1], *state, error_message);
        if (error_message != nullptr && !error_message->empty()) {
          return false;
        }
        result = MakeBitfieldMask64(width, rhs);
        update_scc = false;
        break;
      }
      default:
        if (error_message != nullptr) {
          *error_message = "unsupported compiled scalar pair shift opcode";
        }
        return false;
    }

    if (update_scc) {
      state->scc = result != 0;
    }
    return WriteScalarPairOperand(instruction.operands[0], result, state,
                                  error_message);
  }

  if (instruction.opcode == CompiledOpcode::kSAndB64 ||
      instruction.opcode == CompiledOpcode::kSAndn2B64 ||
      instruction.opcode == CompiledOpcode::kSNandB64 ||
      instruction.opcode == CompiledOpcode::kSOrB64 ||
      instruction.opcode == CompiledOpcode::kSOrn2B64 ||
      instruction.opcode == CompiledOpcode::kSNorB64 ||
      instruction.opcode == CompiledOpcode::kSXorB64 ||
      instruction.opcode == CompiledOpcode::kSXnorB64) {
    const std::uint64_t lhs =
        ReadScalarPairOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint64_t rhs =
        ReadScalarPairOperand(instruction.operands[2], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }

    std::uint64_t result = 0;
    switch (instruction.opcode) {
      case CompiledOpcode::kSAndB64:
        result = lhs & rhs;
        break;
      case CompiledOpcode::kSAndn2B64:
        result = lhs & ~rhs;
        break;
      case CompiledOpcode::kSNandB64:
        result = ~(lhs & rhs);
        break;
      case CompiledOpcode::kSOrB64:
        result = lhs | rhs;
        break;
      case CompiledOpcode::kSOrn2B64:
        result = lhs | ~rhs;
        break;
      case CompiledOpcode::kSNorB64:
        result = ~(lhs | rhs);
        break;
      case CompiledOpcode::kSXorB64:
        result = lhs ^ rhs;
        break;
      case CompiledOpcode::kSXnorB64:
        result = ~(lhs ^ rhs);
        break;
      default:
        if (error_message != nullptr) {
          *error_message = "unsupported compiled scalar pair binary opcode";
        }
        return false;
    }

    state->scc = result != 0;
    return WriteScalarPairOperand(instruction.operands[0], result, state,
                                  error_message);
  }

  const std::uint32_t lhs =
      ReadScalarOperand(instruction.operands[1], *state, error_message);
  if (error_message != nullptr && !error_message->empty()) {
    return false;
  }
  const std::uint32_t rhs =
      ReadScalarOperand(instruction.operands[2], *state, error_message);
  if (error_message != nullptr && !error_message->empty()) {
    return false;
  }

  std::uint32_t result = 0;
  bool scc = state->scc;
  bool update_scc = true;
  switch (instruction.opcode) {
    case CompiledOpcode::kSAddU32: {
      const std::uint64_t wide = static_cast<std::uint64_t>(lhs) + rhs;
      result = static_cast<std::uint32_t>(wide);
      scc = wide > std::numeric_limits<std::uint32_t>::max();
      break;
    }
    case CompiledOpcode::kSAddcU32: {
      const std::uint64_t wide =
          static_cast<std::uint64_t>(lhs) + rhs + (state->scc ? 1u : 0u);
      result = static_cast<std::uint32_t>(wide);
      scc = wide > std::numeric_limits<std::uint32_t>::max();
      break;
    }
    case CompiledOpcode::kSSubU32:
      result = lhs - rhs;
      scc = lhs >= rhs;
      break;
    case CompiledOpcode::kSSubbU32: {
      const std::uint64_t subtrahend =
          static_cast<std::uint64_t>(rhs) + (state->scc ? 1u : 0u);
      result =
          static_cast<std::uint32_t>(static_cast<std::uint64_t>(lhs) - subtrahend);
      scc = static_cast<std::uint64_t>(lhs) >= subtrahend;
      break;
    }
    case CompiledOpcode::kSMinI32: {
      const std::int32_t lhs_signed = BitCast<std::int32_t>(lhs);
      const std::int32_t rhs_signed = BitCast<std::int32_t>(rhs);
      const bool select_lhs = lhs_signed <= rhs_signed;
      result = select_lhs ? lhs : rhs;
      scc = select_lhs;
      break;
    }
    case CompiledOpcode::kSMinU32:
      scc = lhs <= rhs;
      result = scc ? lhs : rhs;
      break;
    case CompiledOpcode::kSMaxI32: {
      const std::int32_t lhs_signed = BitCast<std::int32_t>(lhs);
      const std::int32_t rhs_signed = BitCast<std::int32_t>(rhs);
      const bool select_lhs = lhs_signed >= rhs_signed;
      result = select_lhs ? lhs : rhs;
      scc = select_lhs;
      break;
    }
    case CompiledOpcode::kSMaxU32:
      scc = lhs >= rhs;
      result = scc ? lhs : rhs;
      break;
    case CompiledOpcode::kSMulI32: {
      const std::int64_t product =
          static_cast<std::int64_t>(BitCast<std::int32_t>(lhs)) *
          static_cast<std::int64_t>(BitCast<std::int32_t>(rhs));
      result = static_cast<std::uint32_t>(product);
      update_scc = false;
      break;
    }
    case CompiledOpcode::kSMulHiU32:
      result = static_cast<std::uint32_t>(
          (static_cast<std::uint64_t>(lhs) * static_cast<std::uint64_t>(rhs)) >> 32);
      update_scc = false;
      break;
    case CompiledOpcode::kSMulHiI32: {
      const std::int64_t product =
          static_cast<std::int64_t>(BitCast<std::int32_t>(lhs)) *
          static_cast<std::int64_t>(BitCast<std::int32_t>(rhs));
      result = static_cast<std::uint32_t>(static_cast<std::uint64_t>(product) >> 32);
      update_scc = false;
      break;
    }
    case CompiledOpcode::kSLshl1AddU32:
    case CompiledOpcode::kSLshl2AddU32:
    case CompiledOpcode::kSLshl3AddU32:
    case CompiledOpcode::kSLshl4AddU32: {
      const std::uint32_t shift_amount =
          instruction.opcode == CompiledOpcode::kSLshl1AddU32
              ? 1u
              : (instruction.opcode == CompiledOpcode::kSLshl2AddU32
                     ? 2u
                     : (instruction.opcode == CompiledOpcode::kSLshl3AddU32 ? 3u
                                                                            : 4u));
      const std::uint32_t shifted = lhs << shift_amount;
      const std::uint64_t wide = static_cast<std::uint64_t>(shifted) + rhs;
      result = static_cast<std::uint32_t>(wide);
      scc = wide > std::numeric_limits<std::uint32_t>::max();
      break;
    }
    case CompiledOpcode::kSPackLlB32B16:
      result = (lhs & 0xffffu) | ((rhs & 0xffffu) << 16);
      update_scc = false;
      break;
    case CompiledOpcode::kSPackLhB32B16:
      result = (lhs & 0xffffu) | (rhs & 0xffff0000u);
      update_scc = false;
      break;
    case CompiledOpcode::kSPackHhB32B16:
      result = ((lhs >> 16) & 0xffffu) | (rhs & 0xffff0000u);
      update_scc = false;
      break;
    case CompiledOpcode::kSCselectB32:
      result = state->scc ? lhs : rhs;
      update_scc = false;
      break;
    case CompiledOpcode::kSAbsdiffI32: {
      const std::int64_t diff =
          static_cast<std::int64_t>(BitCast<std::int32_t>(lhs)) -
          static_cast<std::int64_t>(BitCast<std::int32_t>(rhs));
      result = static_cast<std::uint32_t>(diff < 0 ? -diff : diff);
      scc = result != 0;
      break;
    }
    case CompiledOpcode::kSBfeU32:
      result = ExtractUnsignedBitfield32(lhs, rhs);
      scc = result != 0;
      break;
    case CompiledOpcode::kSBfeI32:
      result = ExtractSignedBitfield32(lhs, rhs);
      scc = result != 0;
      break;
    case CompiledOpcode::kSLshlB32:
      result = lhs << (rhs & 31u);
      scc = result != 0;
      break;
    case CompiledOpcode::kSLshrB32:
      result = lhs >> (rhs & 31u);
      scc = result != 0;
      break;
    case CompiledOpcode::kSAshrI32:
      result = static_cast<std::uint32_t>(
          BitCast<std::int32_t>(lhs) >> (rhs & 31u));
      scc = result != 0;
      break;
    case CompiledOpcode::kSBfmB32:
      result = MakeBitfieldMask32(lhs, rhs);
      update_scc = false;
      break;
    case CompiledOpcode::kSAndB32:
      result = lhs & rhs;
      scc = result != 0;
      break;
    case CompiledOpcode::kSAndn2B32:
      result = lhs & ~rhs;
      scc = result != 0;
      break;
    case CompiledOpcode::kSNandB32:
      result = ~(lhs & rhs);
      scc = result != 0;
      break;
    case CompiledOpcode::kSOrB32:
      result = lhs | rhs;
      scc = result != 0;
      break;
    case CompiledOpcode::kSOrn2B32:
      result = lhs | ~rhs;
      scc = result != 0;
      break;
    case CompiledOpcode::kSNorB32:
      result = ~(lhs | rhs);
      scc = result != 0;
      break;
    case CompiledOpcode::kSXorB32:
      result = lhs ^ rhs;
      scc = result != 0;
      break;
    case CompiledOpcode::kSXnorB32:
      result = ~(lhs ^ rhs);
      scc = result != 0;
      break;
    default:
      if (error_message != nullptr) {
        *error_message = "unsupported compiled scalar binary opcode";
      }
      return false;
  }

  if (update_scc) {
    state->scc = scc;
  }
  return WriteScalarOperand(instruction.operands[0], result, state, error_message);
}

bool Gfx950Interpreter::ExecuteScalarCompare(const DecodedInstruction& instruction,
                                             WaveExecutionState* state,
                                             std::string* error_message) const {
  if (!ValidateOperandCount(instruction, 2, error_message)) {
    return false;
  }

  if (instruction.opcode == "S_BITCMP0_B64" || instruction.opcode == "S_BITCMP1_B64") {
    const std::uint64_t lhs =
        ReadScalarPairOperand(instruction.operands[0], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint32_t rhs =
        ReadScalarOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const bool bit_is_set = ((lhs >> (rhs & 63u)) & 1ULL) != 0;
    state->scc = instruction.opcode == "S_BITCMP0_B64" ? !bit_is_set : bit_is_set;
    return true;
  }

  if (instruction.opcode == "S_CMP_EQ_U64" || instruction.opcode == "S_CMP_LG_U64") {
    const std::uint64_t lhs =
        ReadScalarPairOperand(instruction.operands[0], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint64_t rhs =
        ReadScalarPairOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    state->scc = instruction.opcode == "S_CMP_EQ_U64" ? (lhs == rhs) : (lhs != rhs);
    return true;
  }

  const std::uint32_t lhs =
      ReadScalarOperand(instruction.operands[0], *state, error_message);
  if (error_message != nullptr && !error_message->empty()) {
    return false;
  }
  const std::uint32_t rhs =
      ReadScalarOperand(instruction.operands[1], *state, error_message);
  if (error_message != nullptr && !error_message->empty()) {
    return false;
  }

  if (instruction.opcode == "S_CMP_EQ_I32" || instruction.opcode == "S_CMPK_EQ_I32") {
    state->scc = BitCast<std::int32_t>(lhs) == BitCast<std::int32_t>(rhs);
    return true;
  }
  if (instruction.opcode == "S_CMP_LG_I32" || instruction.opcode == "S_CMPK_LG_I32") {
    state->scc = BitCast<std::int32_t>(lhs) != BitCast<std::int32_t>(rhs);
    return true;
  }
  if (instruction.opcode == "S_CMP_GT_I32" || instruction.opcode == "S_CMPK_GT_I32") {
    state->scc = BitCast<std::int32_t>(lhs) > BitCast<std::int32_t>(rhs);
    return true;
  }
  if (instruction.opcode == "S_CMP_GE_I32" || instruction.opcode == "S_CMPK_GE_I32") {
    state->scc = BitCast<std::int32_t>(lhs) >= BitCast<std::int32_t>(rhs);
    return true;
  }
  if (instruction.opcode == "S_CMP_LT_I32" || instruction.opcode == "S_CMPK_LT_I32") {
    state->scc = BitCast<std::int32_t>(lhs) < BitCast<std::int32_t>(rhs);
    return true;
  }
  if (instruction.opcode == "S_CMP_LE_I32" || instruction.opcode == "S_CMPK_LE_I32") {
    state->scc = BitCast<std::int32_t>(lhs) <= BitCast<std::int32_t>(rhs);
    return true;
  }
  if (instruction.opcode == "S_CMP_EQ_U32" || instruction.opcode == "S_CMPK_EQ_U32") {
    state->scc = (lhs == rhs);
    return true;
  }
  if (instruction.opcode == "S_CMP_LG_U32" || instruction.opcode == "S_CMPK_LG_U32") {
    state->scc = (lhs != rhs);
    return true;
  }
  if (instruction.opcode == "S_CMP_GT_U32" || instruction.opcode == "S_CMPK_GT_U32") {
    state->scc = lhs > rhs;
    return true;
  }
  if (instruction.opcode == "S_CMP_GE_U32" || instruction.opcode == "S_CMPK_GE_U32") {
    state->scc = lhs >= rhs;
    return true;
  }
  if (instruction.opcode == "S_CMP_LT_U32" || instruction.opcode == "S_CMPK_LT_U32") {
    state->scc = lhs < rhs;
    return true;
  }
  if (instruction.opcode == "S_CMP_LE_U32" || instruction.opcode == "S_CMPK_LE_U32") {
    state->scc = lhs <= rhs;
    return true;
  }
  if (instruction.opcode == "S_BITCMP0_B32") {
    state->scc = ((lhs >> (rhs & 31u)) & 1u) == 0u;
    return true;
  }
  if (instruction.opcode == "S_BITCMP1_B32") {
    state->scc = ((lhs >> (rhs & 31u)) & 1u) != 0u;
    return true;
  }

  if (error_message != nullptr) {
    *error_message = "unsupported scalar compare opcode";
  }
  return false;
}

bool Gfx950Interpreter::ExecuteScalarCompare(const CompiledInstruction& instruction,
                                             WaveExecutionState* state,
                                             std::string* error_message) const {
  if (!ValidateOperandCount(instruction, 2, error_message)) {
    return false;
  }

  if (instruction.opcode == CompiledOpcode::kSBitcmp0B64 ||
      instruction.opcode == CompiledOpcode::kSBitcmp1B64) {
    const std::uint64_t lhs =
        ReadScalarPairOperand(instruction.operands[0], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint32_t rhs =
        ReadScalarOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const bool bit_is_set = ((lhs >> (rhs & 63u)) & 1ULL) != 0;
    state->scc =
        instruction.opcode == CompiledOpcode::kSBitcmp0B64 ? !bit_is_set
                                                            : bit_is_set;
    return true;
  }

  if (instruction.opcode == CompiledOpcode::kSCmpEqU64 ||
      instruction.opcode == CompiledOpcode::kSCmpLgU64) {
    const std::uint64_t lhs =
        ReadScalarPairOperand(instruction.operands[0], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint64_t rhs =
        ReadScalarPairOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    state->scc = instruction.opcode == CompiledOpcode::kSCmpEqU64 ? (lhs == rhs)
                                                                   : (lhs != rhs);
    return true;
  }

  const std::uint32_t lhs =
      ReadScalarOperand(instruction.operands[0], *state, error_message);
  if (error_message != nullptr && !error_message->empty()) {
    return false;
  }
  const std::uint32_t rhs =
      ReadScalarOperand(instruction.operands[1], *state, error_message);
  if (error_message != nullptr && !error_message->empty()) {
    return false;
  }

  switch (instruction.opcode) {
    case CompiledOpcode::kSCmpEqI32:
      state->scc = BitCast<std::int32_t>(lhs) == BitCast<std::int32_t>(rhs);
      return true;
    case CompiledOpcode::kSCmpLgI32:
      state->scc = BitCast<std::int32_t>(lhs) != BitCast<std::int32_t>(rhs);
      return true;
    case CompiledOpcode::kSCmpGtI32:
      state->scc = BitCast<std::int32_t>(lhs) > BitCast<std::int32_t>(rhs);
      return true;
    case CompiledOpcode::kSCmpGeI32:
      state->scc = BitCast<std::int32_t>(lhs) >= BitCast<std::int32_t>(rhs);
      return true;
    case CompiledOpcode::kSCmpLtI32:
      state->scc = BitCast<std::int32_t>(lhs) < BitCast<std::int32_t>(rhs);
      return true;
    case CompiledOpcode::kSCmpLeI32:
      state->scc = BitCast<std::int32_t>(lhs) <= BitCast<std::int32_t>(rhs);
      return true;
    case CompiledOpcode::kSCmpEqU32:
      state->scc = (lhs == rhs);
      return true;
    case CompiledOpcode::kSCmpLgU32:
      state->scc = (lhs != rhs);
      return true;
    case CompiledOpcode::kSCmpGtU32:
      state->scc = lhs > rhs;
      return true;
    case CompiledOpcode::kSCmpGeU32:
      state->scc = lhs >= rhs;
      return true;
    case CompiledOpcode::kSCmpLtU32:
      state->scc = lhs < rhs;
      return true;
    case CompiledOpcode::kSCmpLeU32:
      state->scc = lhs <= rhs;
      return true;
    case CompiledOpcode::kSBitcmp0B32:
      state->scc = ((lhs >> (rhs & 31u)) & 1u) == 0u;
      return true;
    case CompiledOpcode::kSBitcmp1B32:
      state->scc = ((lhs >> (rhs & 31u)) & 1u) != 0u;
      return true;
    default:
      if (error_message != nullptr) {
        *error_message = "unsupported compiled scalar compare opcode";
      }
      return false;
  }
}

bool Gfx950Interpreter::ExecuteScalarMemory(const DecodedInstruction& instruction,
                                            WaveExecutionState* state,
                                            ExecutionMemory* memory,
                                            std::string* error_message) const {
  if (!ValidateOperandCount(instruction, 3, error_message)) {
    return false;
  }
  if (memory == nullptr) {
    if (error_message != nullptr) {
      *error_message = "memory-backed instruction requires execution memory";
    }
    return false;
  }
  if (instruction.operands[0].kind != OperandKind::kSgpr ||
      instruction.operands[1].kind != OperandKind::kSgpr) {
    if (error_message != nullptr) {
      *error_message = "smem operands must use scalar registers";
    }
    return false;
  }
  if (instruction.operands[1].index + 1 >= state->sgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "smem base register pair out of range";
    }
    return false;
  }

  const std::uint64_t base_address =
      static_cast<std::uint64_t>(state->sgprs[instruction.operands[1].index]) |
      (static_cast<std::uint64_t>(
           state->sgprs[instruction.operands[1].index + 1] & 0xffffu)
       << 32);

  std::uint64_t address = base_address;
  if (instruction.operands[2].kind == OperandKind::kImm32) {
    const std::int64_t signed_address =
        static_cast<std::int64_t>(base_address) +
        static_cast<std::int32_t>(instruction.operands[2].imm32);
    if (signed_address < 0) {
      if (error_message != nullptr) {
        *error_message = "smem address underflow";
      }
      return false;
    }
    address = static_cast<std::uint64_t>(signed_address);
  } else if (instruction.operands[2].kind == OperandKind::kSgpr) {
    if (instruction.operands[2].index >= state->sgprs.size()) {
      if (error_message != nullptr) {
        *error_message = "smem offset register out of range";
      }
      return false;
    }
    address += state->sgprs[instruction.operands[2].index];
  } else {
    if (error_message != nullptr) {
      *error_message = "unsupported smem offset operand";
    }
    return false;
  }

  if (instruction.opcode == "S_LOAD_DWORD") {
    std::uint32_t value = 0;
    if (!ReadMemoryU32(memory, address, &value, error_message)) {
      return false;
    }
    return WriteScalarOperand(instruction.operands[0], value, state, error_message);
  }

  if (instruction.opcode == "S_LOAD_DWORDX2") {
    if (instruction.operands[0].index + 1 >= state->sgprs.size()) {
      if (error_message != nullptr) {
        *error_message = "load dwordx2 destination pair out of range";
      }
      return false;
    }
    std::uint32_t low = 0;
    std::uint32_t high = 0;
    if (!ReadMemoryU32(memory, address, &low, error_message) ||
        !ReadMemoryU32(memory, address + 4, &high, error_message)) {
      return false;
    }
    state->sgprs[instruction.operands[0].index] = low;
    state->sgprs[instruction.operands[0].index + 1] = high;
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }

  if (instruction.opcode == "S_STORE_DWORD") {
    return WriteMemoryU32(memory, address,
                          state->sgprs[instruction.operands[0].index],
                          error_message);
  }

  if (error_message != nullptr) {
    *error_message = "unsupported scalar memory opcode";
  }
  return false;
}

bool Gfx950Interpreter::ExecuteScalarMemory(const CompiledInstruction& instruction,
                                            WaveExecutionState* state,
                                            ExecutionMemory* memory,
                                            std::string* error_message) const {
  if (!ValidateOperandCount(instruction, 3, error_message)) {
    return false;
  }
  if (memory == nullptr) {
    if (error_message != nullptr) {
      *error_message = "memory-backed instruction requires execution memory";
    }
    return false;
  }
  if (instruction.operands[0].kind != OperandKind::kSgpr ||
      instruction.operands[1].kind != OperandKind::kSgpr) {
    if (error_message != nullptr) {
      *error_message = "smem operands must use scalar registers";
    }
    return false;
  }
  if (instruction.operands[1].index + 1 >= state->sgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "smem base register pair out of range";
    }
    return false;
  }

  const std::uint64_t base_address =
      static_cast<std::uint64_t>(state->sgprs[instruction.operands[1].index]) |
      (static_cast<std::uint64_t>(
           state->sgprs[instruction.operands[1].index + 1] & 0xffffu)
       << 32);

  std::uint64_t address = base_address;
  if (instruction.operands[2].kind == OperandKind::kImm32) {
    const std::int64_t signed_address =
        static_cast<std::int64_t>(base_address) +
        static_cast<std::int32_t>(instruction.operands[2].imm32);
    if (signed_address < 0) {
      if (error_message != nullptr) {
        *error_message = "smem address underflow";
      }
      return false;
    }
    address = static_cast<std::uint64_t>(signed_address);
  } else if (instruction.operands[2].kind == OperandKind::kSgpr) {
    if (instruction.operands[2].index >= state->sgprs.size()) {
      if (error_message != nullptr) {
        *error_message = "smem offset register out of range";
      }
      return false;
    }
    address += state->sgprs[instruction.operands[2].index];
  } else {
    if (error_message != nullptr) {
      *error_message = "unsupported smem offset operand";
    }
    return false;
  }

  switch (instruction.opcode) {
    case CompiledOpcode::kSLoadDword: {
      std::uint32_t value = 0;
      if (!ReadMemoryU32(memory, address, &value, error_message)) {
        return false;
      }
      return WriteScalarOperand(instruction.operands[0], value, state, error_message);
    }
    case CompiledOpcode::kSLoadDwordX2: {
      if (instruction.operands[0].index + 1 >= state->sgprs.size()) {
        if (error_message != nullptr) {
          *error_message = "load dwordx2 destination pair out of range";
        }
        return false;
      }
      std::uint32_t low = 0;
      std::uint32_t high = 0;
      if (!ReadMemoryU32(memory, address, &low, error_message) ||
          !ReadMemoryU32(memory, address + 4, &high, error_message)) {
        return false;
      }
      state->sgprs[instruction.operands[0].index] = low;
      state->sgprs[instruction.operands[0].index + 1] = high;
      if (error_message != nullptr) {
        error_message->clear();
      }
      return true;
    }
    case CompiledOpcode::kSStoreDword:
      return WriteMemoryU32(memory, address,
                            state->sgprs[instruction.operands[0].index],
                            error_message);
    default:
      if (error_message != nullptr) {
        *error_message = "unsupported compiled scalar memory opcode";
      }
      return false;
  }
}

bool Gfx950Interpreter::ExecuteVectorMove(const DecodedInstruction& instruction,
                                          WaveExecutionState* state,
                                          std::string* error_message) const {
  if (!ValidateOperandCount(instruction, 2, error_message)) {
    return false;
  }

  if (instruction.opcode == "V_MOV_B64") {
    for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
         ++lane_index) {
      if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
        continue;
      }

      std::uint64_t value = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[1], *state, lane_index,
                                      &value, error_message)) {
        return false;
      }
      if (!WriteVectorPairOperandValue(instruction.operands[0], lane_index, value,
                                       state, error_message)) {
        return false;
      }
    }
    return true;
  }

  for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
       ++lane_index) {
    if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
      continue;
    }

    const std::uint32_t value =
        ReadVectorOperand(instruction.operands[1], *state, lane_index, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    if (!WriteVectorOperand(instruction.operands[0], lane_index, value, state,
                            error_message)) {
      return false;
    }
  }
  return true;
}

bool Gfx950Interpreter::ExecuteVectorMove(const CompiledInstruction& instruction,
                                          WaveExecutionState* state,
                                          std::string* error_message) const {
  if (!ValidateOperandCount(instruction, 2, error_message)) {
    return false;
  }

  if (instruction.opcode == CompiledOpcode::kVMovB64) {
    for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
         ++lane_index) {
      if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
        continue;
      }

      std::uint64_t value = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[1], *state, lane_index,
                                      &value, error_message)) {
        return false;
      }
      if (!WriteVectorPairOperandValue(instruction.operands[0], lane_index, value,
                                       state, error_message)) {
        return false;
      }
    }
    return true;
  }

  for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
       ++lane_index) {
    if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
      continue;
    }

    const std::uint32_t value =
        ReadVectorOperand(instruction.operands[1], *state, lane_index, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    if (!WriteVectorOperand(instruction.operands[0], lane_index, value, state,
                            error_message)) {
      return false;
    }
  }
  return true;
}

bool Gfx950Interpreter::ExecuteVectorToScalar(const DecodedInstruction& instruction,
                                              WaveExecutionState* state,
                                              std::string* error_message) const {
  std::size_t lane_index = 0;
  if (instruction.opcode == "V_READFIRSTLANE_B32") {
    if (!ValidateOperandCount(instruction, 2, error_message)) {
      return false;
    }
    lane_index = FindLowestActiveLane(state->exec_mask);
  } else {
    if (!ValidateOperandCount(instruction, 3, error_message)) {
      return false;
    }
    lane_index = NormalizeWaveLaneIndex(
        ReadScalarOperand(instruction.operands[2], *state, error_message));
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
  }
  const std::uint32_t value =
      lane_index < WaveExecutionState::kLaneCount
          ? ReadVectorOperand(instruction.operands[1], *state, lane_index,
                              error_message)
          : 0u;
  if (error_message != nullptr && !error_message->empty()) {
    return false;
  }
  return WriteScalarOperand(instruction.operands[0], value, state, error_message);
}

bool Gfx950Interpreter::ExecuteVectorToScalar(const CompiledInstruction& instruction,
                                              WaveExecutionState* state,
                                              std::string* error_message) const {
  std::size_t lane_index = 0;
  if (instruction.opcode == CompiledOpcode::kVReadfirstlaneB32) {
    if (!ValidateOperandCount(instruction, 2, error_message)) {
      return false;
    }
    lane_index = FindLowestActiveLane(state->exec_mask);
  } else {
    if (!ValidateOperandCount(instruction, 3, error_message)) {
      return false;
    }
    lane_index = NormalizeWaveLaneIndex(
        ReadScalarOperand(instruction.operands[2], *state, error_message));
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
  }
  const std::uint32_t value =
      lane_index < WaveExecutionState::kLaneCount
          ? ReadVectorOperand(instruction.operands[1], *state, lane_index,
                              error_message)
          : 0u;
  if (error_message != nullptr && !error_message->empty()) {
    return false;
  }
  return WriteScalarOperand(instruction.operands[0], value, state, error_message);
}

bool Gfx950Interpreter::ExecuteVectorUnary(const DecodedInstruction& instruction,
                                           WaveExecutionState* state,
                                           std::string* error_message) const {
  if (!ValidateOperandCount(instruction, 2, error_message)) {
    return false;
  }

  if (instruction.opcode == "V_RCP_F64" || instruction.opcode == "V_RSQ_F64" ||
      instruction.opcode == "V_SQRT_F64" ||
      instruction.opcode == "V_FREXP_MANT_F64" ||
      instruction.opcode == "V_FRACT_F64" || instruction.opcode == "V_TRUNC_F64" ||
      instruction.opcode == "V_CEIL_F64" || instruction.opcode == "V_RNDNE_F64" ||
      instruction.opcode == "V_FLOOR_F64") {
    for (std::size_t lane_index = 0;
         lane_index < WaveExecutionState::kLaneCount; ++lane_index) {
      if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
        continue;
      }

      std::uint64_t value = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[1], *state, lane_index,
                                      &value, error_message)) {
        return false;
      }

      if (!WriteVectorPairOperandValue(
              instruction.operands[0], lane_index,
              EvaluateVectorUnary64To64(instruction.opcode, value), state,
              error_message)) {
        return false;
      }
    }
    return true;
  }

  if (instruction.opcode == "V_CVT_F64_F32" ||
      instruction.opcode == "V_CVT_F64_I32" ||
      instruction.opcode == "V_CVT_F64_U32") {
    for (std::size_t lane_index = 0;
         lane_index < WaveExecutionState::kLaneCount; ++lane_index) {
      if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
        continue;
      }

      const std::uint32_t value = ReadVectorOperand(
          instruction.operands[1], *state, lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }

      if (!WriteVectorPairOperandValue(
              instruction.operands[0], lane_index,
              EvaluateVectorUnary64(instruction.opcode, value), state,
              error_message)) {
        return false;
      }
    }
    return true;
  }

  if (instruction.opcode == "V_CVT_F32_F64" ||
      instruction.opcode == "V_CVT_I32_F64" ||
      instruction.opcode == "V_CVT_U32_F64" ||
      instruction.opcode == "V_FREXP_EXP_I32_F64") {
    for (std::size_t lane_index = 0;
         lane_index < WaveExecutionState::kLaneCount; ++lane_index) {
      if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
        continue;
      }

      std::uint64_t value = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[1], *state, lane_index,
                                      &value, error_message)) {
        return false;
      }

      if (!WriteVectorOperand(
              instruction.operands[0], lane_index,
              EvaluateVectorUnaryFrom64(instruction.opcode, value), state,
              error_message)) {
        return false;
      }
    }
    return true;
  }

  for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
       ++lane_index) {
    if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
      continue;
    }

    const std::uint32_t value =
        ReadVectorOperand(instruction.operands[1], *state, lane_index, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }

    if (instruction.opcode != "V_NOT_B32" &&
        instruction.opcode != "V_BFREV_B32" &&
        instruction.opcode != "V_FFBH_U32" &&
        instruction.opcode != "V_FFBL_B32" &&
        instruction.opcode != "V_FFBH_I32" &&
        instruction.opcode != "V_CVT_F16_U16" &&
        instruction.opcode != "V_CVT_F16_I16" &&
        instruction.opcode != "V_CVT_U16_F16" &&
        instruction.opcode != "V_CVT_I16_F16" &&
        instruction.opcode != "V_SAT_PK_U8_I16" &&
        instruction.opcode != "V_CVT_F32_UBYTE0" &&
        instruction.opcode != "V_CVT_F32_UBYTE1" &&
        instruction.opcode != "V_CVT_F32_UBYTE2" &&
        instruction.opcode != "V_CVT_F32_UBYTE3" &&
        instruction.opcode != "V_CVT_F32_I32" &&
        instruction.opcode != "V_CVT_F32_U32" &&
        instruction.opcode != "V_CVT_U32_F32" &&
        instruction.opcode != "V_CVT_I32_F32" &&
        instruction.opcode != "V_CVT_RPI_I32_F32" &&
        instruction.opcode != "V_CVT_FLR_I32_F32" &&
        instruction.opcode != "V_CVT_F16_F32" &&
        instruction.opcode != "V_CVT_F32_F16" &&
        instruction.opcode != "V_RCP_F16" &&
        instruction.opcode != "V_SQRT_F16" &&
        instruction.opcode != "V_RSQ_F16" &&
        instruction.opcode != "V_LOG_F16" &&
        instruction.opcode != "V_EXP_F16" &&
        instruction.opcode != "V_SIN_F16" &&
        instruction.opcode != "V_COS_F16" &&
        instruction.opcode != "V_FREXP_MANT_F16" &&
        instruction.opcode != "V_FREXP_EXP_I16_F16" &&
        instruction.opcode != "V_FRACT_F16" &&
        instruction.opcode != "V_TRUNC_F16" &&
        instruction.opcode != "V_CEIL_F16" &&
        instruction.opcode != "V_RNDNE_F16" &&
        instruction.opcode != "V_FLOOR_F16" &&
        instruction.opcode != "V_EXP_F32" &&
        instruction.opcode != "V_EXP_LEGACY_F32" &&
        instruction.opcode != "V_LOG_F32" &&
        instruction.opcode != "V_LOG_LEGACY_F32" &&
        instruction.opcode != "V_RCP_F32" &&
        instruction.opcode != "V_RCP_IFLAG_F32" &&
        instruction.opcode != "V_RSQ_F32" &&
        instruction.opcode != "V_SQRT_F32" &&
        instruction.opcode != "V_SIN_F32" &&
        instruction.opcode != "V_COS_F32" &&
        instruction.opcode != "V_FREXP_EXP_I32_F32" &&
        instruction.opcode != "V_FREXP_MANT_F32" &&
        instruction.opcode != "V_FRACT_F32" &&
        instruction.opcode != "V_TRUNC_F32" &&
        instruction.opcode != "V_CEIL_F32" &&
        instruction.opcode != "V_RNDNE_F32" &&
        instruction.opcode != "V_FLOOR_F32") {
      if (error_message != nullptr) {
        *error_message = "unsupported vector unary opcode";
      }
      return false;
    }

    if (!WriteVectorOperand(instruction.operands[0], lane_index,
                            EvaluateVectorUnary32(instruction.opcode, value),
                            state, error_message)) {
      return false;
    }
  }
  return true;
}

bool Gfx950Interpreter::ExecuteVectorUnary(const CompiledInstruction& instruction,
                                           WaveExecutionState* state,
                                           std::string* error_message) const {
  if (!ValidateOperandCount(instruction, 2, error_message)) {
    return false;
  }

  if (instruction.opcode == CompiledOpcode::kVRcpF64 ||
      instruction.opcode == CompiledOpcode::kVRsqF64 ||
      instruction.opcode == CompiledOpcode::kVSqrtF64 ||
      instruction.opcode == CompiledOpcode::kVFrexpMantF64 ||
      instruction.opcode == CompiledOpcode::kVFractF64 ||
      instruction.opcode == CompiledOpcode::kVTruncF64 ||
      instruction.opcode == CompiledOpcode::kVCeilF64 ||
      instruction.opcode == CompiledOpcode::kVRndneF64 ||
      instruction.opcode == CompiledOpcode::kVFloorF64) {
    for (std::size_t lane_index = 0;
         lane_index < WaveExecutionState::kLaneCount; ++lane_index) {
      if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
        continue;
      }

      std::uint64_t value = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[1], *state, lane_index,
                                      &value, error_message)) {
        return false;
      }

      if (!WriteVectorPairOperandValue(
              instruction.operands[0], lane_index,
              EvaluateVectorUnary64To64(instruction.opcode, value), state,
              error_message)) {
        return false;
      }
    }
    return true;
  }

  if (instruction.opcode == CompiledOpcode::kVCvtF64F32 ||
      instruction.opcode == CompiledOpcode::kVCvtF64I32 ||
      instruction.opcode == CompiledOpcode::kVCvtF64U32) {
    for (std::size_t lane_index = 0;
         lane_index < WaveExecutionState::kLaneCount; ++lane_index) {
      if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
        continue;
      }

      const std::uint32_t value = ReadVectorOperand(
          instruction.operands[1], *state, lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }

      if (!WriteVectorPairOperandValue(
              instruction.operands[0], lane_index,
              EvaluateVectorUnary64(instruction.opcode, value), state,
              error_message)) {
        return false;
      }
    }
    return true;
  }

  if (instruction.opcode == CompiledOpcode::kVCvtF32F64 ||
      instruction.opcode == CompiledOpcode::kVCvtI32F64 ||
      instruction.opcode == CompiledOpcode::kVCvtU32F64 ||
      instruction.opcode == CompiledOpcode::kVFrexpExpI32F64) {
    for (std::size_t lane_index = 0;
         lane_index < WaveExecutionState::kLaneCount; ++lane_index) {
      if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
        continue;
      }

      std::uint64_t value = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[1], *state, lane_index,
                                      &value, error_message)) {
        return false;
      }

      if (!WriteVectorOperand(
              instruction.operands[0], lane_index,
              EvaluateVectorUnaryFrom64(instruction.opcode, value), state,
              error_message)) {
        return false;
      }
    }
    return true;
  }

  for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
       ++lane_index) {
    if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
      continue;
    }

    const std::uint32_t value =
        ReadVectorOperand(instruction.operands[1], *state, lane_index, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }

    switch (instruction.opcode) {
      case CompiledOpcode::kVNotB32:
      case CompiledOpcode::kVBfrevB32:
      case CompiledOpcode::kVFfbhU32:
      case CompiledOpcode::kVFfblB32:
      case CompiledOpcode::kVFfbhI32:
      case CompiledOpcode::kVCvtF16U16:
      case CompiledOpcode::kVCvtF16I16:
      case CompiledOpcode::kVCvtU16F16:
      case CompiledOpcode::kVCvtI16F16:
      case CompiledOpcode::kVSatPkU8I16:
      case CompiledOpcode::kVCvtF32Ubyte0:
      case CompiledOpcode::kVCvtF32Ubyte1:
      case CompiledOpcode::kVCvtF32Ubyte2:
      case CompiledOpcode::kVCvtF32Ubyte3:
      case CompiledOpcode::kVCvtF32I32:
      case CompiledOpcode::kVCvtF32U32:
      case CompiledOpcode::kVCvtU32F32:
      case CompiledOpcode::kVCvtI32F32:
      case CompiledOpcode::kVCvtRpiI32F32:
      case CompiledOpcode::kVCvtFlrI32F32:
      case CompiledOpcode::kVCvtF16F32:
      case CompiledOpcode::kVCvtF32F16:
      case CompiledOpcode::kVRcpF16:
      case CompiledOpcode::kVSqrtF16:
      case CompiledOpcode::kVRsqF16:
      case CompiledOpcode::kVLogF16:
      case CompiledOpcode::kVExpF16:
      case CompiledOpcode::kVSinF16:
      case CompiledOpcode::kVCosF16:
      case CompiledOpcode::kVFrexpMantF16:
      case CompiledOpcode::kVFrexpExpI16F16:
      case CompiledOpcode::kVFractF16:
      case CompiledOpcode::kVTruncF16:
      case CompiledOpcode::kVCeilF16:
      case CompiledOpcode::kVRndneF16:
      case CompiledOpcode::kVFloorF16:
      case CompiledOpcode::kVExpF32:
      case CompiledOpcode::kVExpLegacyF32:
      case CompiledOpcode::kVLogF32:
      case CompiledOpcode::kVLogLegacyF32:
      case CompiledOpcode::kVRcpF32:
      case CompiledOpcode::kVRcpIflagF32:
      case CompiledOpcode::kVRsqF32:
      case CompiledOpcode::kVSqrtF32:
      case CompiledOpcode::kVSinF32:
      case CompiledOpcode::kVCosF32:
      case CompiledOpcode::kVFrexpExpI32F32:
      case CompiledOpcode::kVFrexpMantF32:
      case CompiledOpcode::kVFractF32:
      case CompiledOpcode::kVTruncF32:
      case CompiledOpcode::kVCeilF32:
      case CompiledOpcode::kVRndneF32:
      case CompiledOpcode::kVFloorF32:
        break;
      default:
        if (error_message != nullptr) {
          *error_message = "unsupported compiled vector unary opcode";
        }
        return false;
    }

    if (!WriteVectorOperand(instruction.operands[0], lane_index,
                            EvaluateVectorUnary32(instruction.opcode, value),
                            state, error_message)) {
      return false;
    }
  }
  return true;
}

bool Gfx950Interpreter::ExecuteVectorBinary(const DecodedInstruction& instruction,
                                            WaveExecutionState* state,
                                            std::string* error_message) const {
  if (IsVectorCarryOutBinaryOpcode(instruction.opcode)) {
    if (!ValidateOperandCount(instruction, 4, error_message)) {
      return false;
    }
    std::uint64_t mask = state->vcc_mask;
    for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
         ++lane_index) {
      if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
        continue;
      }
      const std::uint32_t lhs = ReadVectorOperand(instruction.operands[2], *state,
                                                  lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      const std::uint32_t rhs = ReadVectorOperand(instruction.operands[3], *state,
                                                  lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      std::uint32_t result = 0;
      bool carry = false;
      if (instruction.opcode == "V_ADD_CO_U32") {
        const std::uint64_t full = static_cast<std::uint64_t>(lhs) +
                                   static_cast<std::uint64_t>(rhs);
        result = static_cast<std::uint32_t>(full);
        carry = (full >> 32) != 0;
      } else if (instruction.opcode == "V_SUB_CO_U32") {
        result = lhs - rhs;
        carry = lhs < rhs;
      } else {
        result = rhs - lhs;
        carry = rhs < lhs;
      }
      if (!WriteVectorOperand(instruction.operands[0], lane_index, result, state,
                              error_message)) {
        return false;
      }
      if (carry) {
        mask |= (1ULL << lane_index);
      } else {
        mask &= ~(1ULL << lane_index);
      }
    }
    state->vcc_mask = mask;
    return WriteScalarPairOperand(instruction.operands[1], mask, state,
                                  error_message);
  }

  if (IsVectorCarryInBinaryOpcode(instruction.opcode)) {
    if (!ValidateOperandCount(instruction, 5, error_message)) {
      return false;
    }
    const std::uint64_t carry_in_mask =
        ReadScalarPairOperand(instruction.operands[4], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    std::uint64_t mask = state->vcc_mask;
    for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
         ++lane_index) {
      if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
        continue;
      }
      const std::uint32_t lhs = ReadVectorOperand(instruction.operands[2], *state,
                                                  lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      const std::uint32_t rhs = ReadVectorOperand(instruction.operands[3], *state,
                                                  lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      const std::uint32_t carry_in =
          static_cast<std::uint32_t>((carry_in_mask >> lane_index) & 1ULL);
      std::uint32_t result = 0;
      bool carry = false;
      if (instruction.opcode == "V_ADDC_CO_U32") {
        const std::uint64_t full = static_cast<std::uint64_t>(lhs) +
                                   static_cast<std::uint64_t>(rhs) + carry_in;
        result = static_cast<std::uint32_t>(full);
        carry = (full >> 32) != 0;
      } else if (instruction.opcode == "V_SUBB_CO_U32") {
        const std::uint64_t subtrahend =
            static_cast<std::uint64_t>(rhs) + carry_in;
        result = static_cast<std::uint32_t>(static_cast<std::uint64_t>(lhs) -
                                            subtrahend);
        carry = static_cast<std::uint64_t>(lhs) < subtrahend;
      } else {
        const std::uint64_t subtrahend =
            static_cast<std::uint64_t>(lhs) + carry_in;
        result = static_cast<std::uint32_t>(static_cast<std::uint64_t>(rhs) -
                                            subtrahend);
        carry = static_cast<std::uint64_t>(rhs) < subtrahend;
      }
      if (!WriteVectorOperand(instruction.operands[0], lane_index, result, state,
                              error_message)) {
        return false;
      }
      if (carry) {
        mask |= (1ULL << lane_index);
      } else {
        mask &= ~(1ULL << lane_index);
      }
    }
    state->vcc_mask = mask;
    return WriteScalarPairOperand(instruction.operands[1], mask, state,
                                  error_message);
  }

  if (!ValidateOperandCount(instruction, 3, error_message)) {
    return false;
  }

  if (instruction.opcode == "V_WRITELANE_B32") {
    const std::uint32_t value =
        ReadScalarOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::size_t lane_index = NormalizeWaveLaneIndex(
        ReadScalarOperand(instruction.operands[2], *state, error_message));
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    return WriteVectorOperand(instruction.operands[0], lane_index, value, state,
                              error_message);
  }

  for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
       ++lane_index) {
    if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
      continue;
    }

    if (instruction.opcode == "V_LSHLREV_B64" ||
        instruction.opcode == "V_LSHRREV_B64" ||
        instruction.opcode == "V_ASHRREV_I64") {
      const std::uint32_t shift_count = ReadVectorOperand(
          instruction.operands[1], *state, lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      std::uint64_t source = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[2], *state, lane_index,
                                      &source, error_message)) {
        return false;
      }
      std::uint64_t result = 0;
      if (instruction.opcode == "V_LSHLREV_B64") {
        result = source << (shift_count & 63u);
      } else if (instruction.opcode == "V_LSHRREV_B64") {
        result = source >> (shift_count & 63u);
      } else {
        result = static_cast<std::uint64_t>(
            static_cast<std::int64_t>(source) >> (shift_count & 63u));
      }
      if (!WriteVectorPairOperandValue(instruction.operands[0], lane_index, result,
                                       state, error_message)) {
        return false;
      }
      continue;
    }

    if (instruction.opcode == "V_ADD_F64" || instruction.opcode == "V_MUL_F64" ||
        instruction.opcode == "V_MIN_F64" || instruction.opcode == "V_MAX_F64") {
      std::uint64_t lhs = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[1], *state, lane_index,
                                      &lhs, error_message)) {
        return false;
      }
      std::uint64_t rhs = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[2], *state, lane_index,
                                      &rhs, error_message)) {
        return false;
      }
      const std::uint64_t result =
          EvaluateVectorFloatBinaryF64(instruction.opcode, lhs, rhs);
      if (!WriteVectorPairOperandValue(instruction.operands[0], lane_index, result,
                                       state, error_message)) {
        return false;
      }
      continue;
    }

    const std::uint32_t lhs =
        ReadVectorOperand(instruction.operands[1], *state, lane_index, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint32_t rhs =
        ReadVectorOperand(instruction.operands[2], *state, lane_index, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }

    std::uint32_t result = 0;
    if (instruction.opcode == "V_CNDMASK_B32") {
      result = ((state->vcc_mask >> lane_index) & 1ULL) != 0 ? rhs : lhs;
    } else if (instruction.opcode == "V_ADD_U32") {
      result = lhs + rhs;
    } else if (instruction.opcode == "V_ADD_F16" ||
               instruction.opcode == "V_SUB_F16" ||
               instruction.opcode == "V_MUL_F16" ||
               instruction.opcode == "V_MIN_F16" ||
               instruction.opcode == "V_MAX_F16") {
      result = EvaluateVectorFloatBinaryF16(instruction.opcode, lhs, rhs);
    } else if (instruction.opcode == "V_ADD_F32" ||
               instruction.opcode == "V_SUB_F32" ||
               instruction.opcode == "V_MUL_F32" ||
               instruction.opcode == "V_MIN_F32" ||
               instruction.opcode == "V_MAX_F32") {
      result = EvaluateVectorFloatBinaryF32(instruction.opcode, lhs, rhs);
    } else if (instruction.opcode == "V_SUB_U32") {
      result = lhs - rhs;
    } else if (instruction.opcode == "V_SUBREV_U32") {
      result = rhs - lhs;
    } else if (instruction.opcode == "V_MUL_LO_U32") {
      result = static_cast<std::uint32_t>(
          static_cast<std::uint64_t>(lhs) * static_cast<std::uint64_t>(rhs));
    } else if (instruction.opcode == "V_MUL_HI_U32") {
      result = static_cast<std::uint32_t>(
          (static_cast<std::uint64_t>(lhs) * static_cast<std::uint64_t>(rhs)) >>
          32);
    } else if (instruction.opcode == "V_MUL_HI_I32") {
      const std::int64_t product =
          static_cast<std::int64_t>(BitCast<std::int32_t>(lhs)) *
          static_cast<std::int64_t>(BitCast<std::int32_t>(rhs));
      result = static_cast<std::uint32_t>(product >> 32);
    } else if (instruction.opcode == "V_BCNT_U32_B32") {
      result = static_cast<std::uint32_t>(__builtin_popcount(lhs)) + rhs;
    } else if (instruction.opcode == "V_BFM_B32") {
      result = MakeBitfieldMask32(lhs, rhs);
    } else if (instruction.opcode == "V_MBCNT_LO_U32_B32") {
      const std::uint32_t prefix_width =
          lane_index < 32u ? static_cast<std::uint32_t>(lane_index) : 32u;
      result = CountLowBits(lhs, prefix_width) + rhs;
    } else if (instruction.opcode == "V_MBCNT_HI_U32_B32") {
      const std::uint32_t prefix_width =
          lane_index < 32u ? 0u : static_cast<std::uint32_t>(lane_index - 32u);
      result = CountLowBits(lhs, prefix_width) + rhs;
    } else if (instruction.opcode == "V_MIN_I32") {
      result = static_cast<std::uint32_t>(std::min(BitCast<std::int32_t>(lhs),
                                                   BitCast<std::int32_t>(rhs)));
    } else if (instruction.opcode == "V_MAX_I32") {
      result = static_cast<std::uint32_t>(std::max(BitCast<std::int32_t>(lhs),
                                                   BitCast<std::int32_t>(rhs)));
    } else if (instruction.opcode == "V_MIN_U32") {
      result = std::min(lhs, rhs);
    } else if (instruction.opcode == "V_MAX_U32") {
      result = std::max(lhs, rhs);
    } else if (instruction.opcode == "V_LSHRREV_B32") {
      result = rhs >> (lhs & 31u);
    } else if (instruction.opcode == "V_ASHRREV_I32") {
      result = static_cast<std::uint32_t>(BitCast<std::int32_t>(rhs) >>
                                          (lhs & 31u));
    } else if (instruction.opcode == "V_LSHLREV_B32") {
      result = rhs << (lhs & 31u);
    } else if (instruction.opcode == "V_AND_B32") {
      result = lhs & rhs;
    } else if (instruction.opcode == "V_OR_B32") {
      result = lhs | rhs;
    } else if (instruction.opcode == "V_XOR_B32") {
      result = lhs ^ rhs;
    } else {
      if (error_message != nullptr) {
        *error_message = "unsupported vector binary opcode";
      }
      return false;
    }

    if (!WriteVectorOperand(instruction.operands[0], lane_index, result, state,
                            error_message)) {
      return false;
    }
  }
  return true;
}

bool Gfx950Interpreter::ExecuteVectorBinary(const CompiledInstruction& instruction,
                                            WaveExecutionState* state,
                                            std::string* error_message) const {
  if (instruction.opcode == CompiledOpcode::kVAddCoU32 ||
      instruction.opcode == CompiledOpcode::kVSubCoU32 ||
      instruction.opcode == CompiledOpcode::kVSubRevCoU32) {
    if (!ValidateOperandCount(instruction, 4, error_message)) {
      return false;
    }
    std::uint64_t mask = state->vcc_mask;
    for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
         ++lane_index) {
      if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
        continue;
      }
      const std::uint32_t lhs = ReadVectorOperand(instruction.operands[2], *state,
                                                  lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      const std::uint32_t rhs = ReadVectorOperand(instruction.operands[3], *state,
                                                  lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      std::uint32_t result = 0;
      bool carry = false;
      if (instruction.opcode == CompiledOpcode::kVAddCoU32) {
        const std::uint64_t full = static_cast<std::uint64_t>(lhs) +
                                   static_cast<std::uint64_t>(rhs);
        result = static_cast<std::uint32_t>(full);
        carry = (full >> 32) != 0;
      } else if (instruction.opcode == CompiledOpcode::kVSubCoU32) {
        result = lhs - rhs;
        carry = lhs < rhs;
      } else {
        result = rhs - lhs;
        carry = rhs < lhs;
      }
      if (!WriteVectorOperand(instruction.operands[0], lane_index, result, state,
                              error_message)) {
        return false;
      }
      if (carry) {
        mask |= (1ULL << lane_index);
      } else {
        mask &= ~(1ULL << lane_index);
      }
    }
    state->vcc_mask = mask;
    return WriteScalarPairOperand(instruction.operands[1], mask, state,
                                  error_message);
  }

  if (instruction.opcode == CompiledOpcode::kVAddcCoU32 ||
      instruction.opcode == CompiledOpcode::kVSubbCoU32 ||
      instruction.opcode == CompiledOpcode::kVSubbrevCoU32) {
    if (!ValidateOperandCount(instruction, 5, error_message)) {
      return false;
    }
    const std::uint64_t carry_in_mask =
        ReadScalarPairOperand(instruction.operands[4], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    std::uint64_t mask = state->vcc_mask;
    for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
         ++lane_index) {
      if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
        continue;
      }
      const std::uint32_t lhs = ReadVectorOperand(instruction.operands[2], *state,
                                                  lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      const std::uint32_t rhs = ReadVectorOperand(instruction.operands[3], *state,
                                                  lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      const std::uint32_t carry_in =
          static_cast<std::uint32_t>((carry_in_mask >> lane_index) & 1ULL);
      std::uint32_t result = 0;
      bool carry = false;
      if (instruction.opcode == CompiledOpcode::kVAddcCoU32) {
        const std::uint64_t full = static_cast<std::uint64_t>(lhs) +
                                   static_cast<std::uint64_t>(rhs) + carry_in;
        result = static_cast<std::uint32_t>(full);
        carry = (full >> 32) != 0;
      } else if (instruction.opcode == CompiledOpcode::kVSubbCoU32) {
        const std::uint64_t subtrahend =
            static_cast<std::uint64_t>(rhs) + carry_in;
        result = static_cast<std::uint32_t>(static_cast<std::uint64_t>(lhs) -
                                            subtrahend);
        carry = static_cast<std::uint64_t>(lhs) < subtrahend;
      } else {
        const std::uint64_t subtrahend =
            static_cast<std::uint64_t>(lhs) + carry_in;
        result = static_cast<std::uint32_t>(static_cast<std::uint64_t>(rhs) -
                                            subtrahend);
        carry = static_cast<std::uint64_t>(rhs) < subtrahend;
      }
      if (!WriteVectorOperand(instruction.operands[0], lane_index, result, state,
                              error_message)) {
        return false;
      }
      if (carry) {
        mask |= (1ULL << lane_index);
      } else {
        mask &= ~(1ULL << lane_index);
      }
    }
    state->vcc_mask = mask;
    return WriteScalarPairOperand(instruction.operands[1], mask, state,
                                  error_message);
  }

  if (!ValidateOperandCount(instruction, 3, error_message)) {
    return false;
  }

  if (instruction.opcode == CompiledOpcode::kVWritelaneB32) {
    const std::uint32_t value =
        ReadScalarOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::size_t lane_index = NormalizeWaveLaneIndex(
        ReadScalarOperand(instruction.operands[2], *state, error_message));
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    return WriteVectorOperand(instruction.operands[0], lane_index, value, state,
                              error_message);
  }

  for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
       ++lane_index) {
    if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
      continue;
    }

    if (instruction.opcode == CompiledOpcode::kVLshlRevB64 ||
        instruction.opcode == CompiledOpcode::kVLshrRevB64 ||
        instruction.opcode == CompiledOpcode::kVAshrRevI64) {
      const std::uint32_t shift_count = ReadVectorOperand(
          instruction.operands[1], *state, lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      std::uint64_t source = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[2], *state, lane_index,
                                      &source, error_message)) {
        return false;
      }
      std::uint64_t result = 0;
      if (instruction.opcode == CompiledOpcode::kVLshlRevB64) {
        result = source << (shift_count & 63u);
      } else if (instruction.opcode == CompiledOpcode::kVLshrRevB64) {
        result = source >> (shift_count & 63u);
      } else {
        result = static_cast<std::uint64_t>(
            static_cast<std::int64_t>(source) >> (shift_count & 63u));
      }
      if (!WriteVectorPairOperandValue(instruction.operands[0], lane_index, result,
                                       state, error_message)) {
        return false;
      }
      continue;
    }

    if (instruction.opcode == CompiledOpcode::kVAddF64 ||
        instruction.opcode == CompiledOpcode::kVMulF64 ||
        instruction.opcode == CompiledOpcode::kVMinF64 ||
        instruction.opcode == CompiledOpcode::kVMaxF64) {
      std::uint64_t lhs = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[1], *state, lane_index,
                                      &lhs, error_message)) {
        return false;
      }
      std::uint64_t rhs = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[2], *state, lane_index,
                                      &rhs, error_message)) {
        return false;
      }
      const std::uint64_t result =
          EvaluateVectorFloatBinaryF64(instruction.opcode, lhs, rhs);
      if (!WriteVectorPairOperandValue(instruction.operands[0], lane_index, result,
                                       state, error_message)) {
        return false;
      }
      continue;
    }

    const std::uint32_t lhs =
        ReadVectorOperand(instruction.operands[1], *state, lane_index, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint32_t rhs =
        ReadVectorOperand(instruction.operands[2], *state, lane_index, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }

    std::uint32_t result = 0;
    switch (instruction.opcode) {
      case CompiledOpcode::kVCndmaskB32:
        result = ((state->vcc_mask >> lane_index) & 1ULL) != 0 ? rhs : lhs;
        break;
      case CompiledOpcode::kVAddU32:
        result = lhs + rhs;
        break;
      case CompiledOpcode::kVAddF16:
      case CompiledOpcode::kVSubF16:
      case CompiledOpcode::kVMulF16:
      case CompiledOpcode::kVMinF16:
      case CompiledOpcode::kVMaxF16:
        result = EvaluateVectorFloatBinaryF16(instruction.opcode, lhs, rhs);
        break;
      case CompiledOpcode::kVAddF32:
      case CompiledOpcode::kVSubF32:
      case CompiledOpcode::kVMulF32:
      case CompiledOpcode::kVMinF32:
      case CompiledOpcode::kVMaxF32:
        result = EvaluateVectorFloatBinaryF32(instruction.opcode, lhs, rhs);
        break;
      case CompiledOpcode::kVSubU32:
        result = lhs - rhs;
        break;
      case CompiledOpcode::kVSubRevU32:
        result = rhs - lhs;
        break;
      case CompiledOpcode::kVMulLoU32:
        result = static_cast<std::uint32_t>(
            static_cast<std::uint64_t>(lhs) * static_cast<std::uint64_t>(rhs));
        break;
      case CompiledOpcode::kVMulHiU32:
        result = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(lhs) * static_cast<std::uint64_t>(rhs)) >>
            32);
        break;
      case CompiledOpcode::kVMulHiI32: {
        const std::int64_t product =
            static_cast<std::int64_t>(BitCast<std::int32_t>(lhs)) *
            static_cast<std::int64_t>(BitCast<std::int32_t>(rhs));
        result = static_cast<std::uint32_t>(product >> 32);
        break;
      }
      case CompiledOpcode::kVBcntU32B32:
        result = static_cast<std::uint32_t>(__builtin_popcount(lhs)) + rhs;
        break;
      case CompiledOpcode::kVBfmB32:
        result = MakeBitfieldMask32(lhs, rhs);
        break;
      case CompiledOpcode::kVMbcntLoU32B32: {
        const std::uint32_t prefix_width =
            lane_index < 32u ? static_cast<std::uint32_t>(lane_index) : 32u;
        result = CountLowBits(lhs, prefix_width) + rhs;
        break;
      }
      case CompiledOpcode::kVMbcntHiU32B32: {
        const std::uint32_t prefix_width =
            lane_index < 32u ? 0u : static_cast<std::uint32_t>(lane_index - 32u);
        result = CountLowBits(lhs, prefix_width) + rhs;
        break;
      }
      case CompiledOpcode::kVMinI32:
        result = static_cast<std::uint32_t>(
            std::min(BitCast<std::int32_t>(lhs), BitCast<std::int32_t>(rhs)));
        break;
      case CompiledOpcode::kVMaxI32:
        result = static_cast<std::uint32_t>(
            std::max(BitCast<std::int32_t>(lhs), BitCast<std::int32_t>(rhs)));
        break;
      case CompiledOpcode::kVMinU32:
        result = std::min(lhs, rhs);
        break;
      case CompiledOpcode::kVMaxU32:
        result = std::max(lhs, rhs);
        break;
      case CompiledOpcode::kVMinF64:
      case CompiledOpcode::kVMaxF64:
        if (error_message != nullptr) {
          *error_message = "64-bit float vector binary opcode reached 32-bit path";
        }
        return false;
      case CompiledOpcode::kVLshrRevB32:
        result = rhs >> (lhs & 31u);
        break;
      case CompiledOpcode::kVAshrRevI32:
        result = static_cast<std::uint32_t>(BitCast<std::int32_t>(rhs) >>
                                            (lhs & 31u));
        break;
      case CompiledOpcode::kVLshlRevB32:
        result = rhs << (lhs & 31u);
        break;
      case CompiledOpcode::kVAndB32:
        result = lhs & rhs;
        break;
      case CompiledOpcode::kVOrB32:
        result = lhs | rhs;
        break;
      case CompiledOpcode::kVXorB32:
        result = lhs ^ rhs;
        break;
      default:
        if (error_message != nullptr) {
          *error_message = "unsupported compiled vector binary opcode";
        }
        return false;
    }

    if (!WriteVectorOperand(instruction.operands[0], lane_index, result, state,
                            error_message)) {
      return false;
    }
  }
  return true;
}

bool Gfx950Interpreter::ExecuteVectorTernary(const DecodedInstruction& instruction,
                                             WaveExecutionState* state,
                                             std::string* error_message) const {
  if ((instruction.opcode == "V_MAD_U64_U32" ||
       instruction.opcode == "V_MAD_I64_I32")) {
    if (!ValidateOperandCount(instruction, 5, error_message)) {
      return false;
    }
  } else if (!ValidateOperandCount(instruction, 4, error_message)) {
    return false;
  }

  for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
       ++lane_index) {
    if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
      continue;
    }

    if (instruction.opcode == "V_LSHL_ADD_U64") {
      std::uint64_t src0 = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[1], *state, lane_index,
                                      &src0, error_message)) {
        return false;
      }
      const std::uint32_t shift_count =
          ReadVectorOperand(instruction.operands[2], *state, lane_index,
                            error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      std::uint64_t src2 = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[3], *state, lane_index,
                                      &src2, error_message)) {
        return false;
      }
      const std::uint64_t result = (src0 << (shift_count & 63u)) + src2;
      if (!WriteVectorPairOperandValue(instruction.operands[0], lane_index, result,
                                       state, error_message)) {
        return false;
      }
      continue;
    }

    if (instruction.opcode == "V_MAD_U64_U32" ||
        instruction.opcode == "V_MAD_I64_I32") {
      const std::uint32_t src0 =
          ReadVectorOperand(instruction.operands[2], *state, lane_index,
                            error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      const std::uint32_t src1 =
          ReadVectorOperand(instruction.operands[3], *state, lane_index,
                            error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      std::uint64_t src2 = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[4], *state, lane_index,
                                      &src2, error_message)) {
        return false;
      }

      std::uint64_t result = 0;
      bool carry_or_overflow = false;
      if (instruction.opcode == "V_MAD_U64_U32") {
        const unsigned __int128 full =
            static_cast<unsigned __int128>(src0) *
                static_cast<unsigned __int128>(src1) +
            static_cast<unsigned __int128>(src2);
        result = static_cast<std::uint64_t>(full);
        carry_or_overflow = (full >> 64) != 0;
      } else {
        const __int128 full =
            static_cast<__int128>(static_cast<std::int64_t>(
                BitCast<std::int32_t>(src0))) *
                static_cast<__int128>(static_cast<std::int64_t>(
                    BitCast<std::int32_t>(src1))) +
            static_cast<__int128>(BitCast<std::int64_t>(src2));
        result = static_cast<std::uint64_t>(full);
        carry_or_overflow =
            full > static_cast<__int128>(std::numeric_limits<std::int64_t>::max()) ||
            full < static_cast<__int128>(std::numeric_limits<std::int64_t>::min());
      }

      if (!WriteVectorPairOperandValue(instruction.operands[0], lane_index, result,
                                       state, error_message)) {
        return false;
      }
      if (instruction.operands[1].kind == OperandKind::kSgpr &&
          instruction.operands[1].index + 1 < state->sgprs.size()) {
        std::uint64_t mask =
            ReadScalarPairOperand(instruction.operands[1], *state, error_message);
        if (error_message != nullptr && !error_message->empty()) {
          return false;
        }
        if (carry_or_overflow) {
          mask |= (1ULL << lane_index);
        } else {
          mask &= ~(1ULL << lane_index);
        }
        if (!WriteScalarPairOperand(instruction.operands[1], mask, state,
                                    error_message)) {
          return false;
        }
        if (instruction.operands[1].index == kVccPairSgprIndex) {
          state->vcc_mask = mask;
        }
      }
      continue;
    }

    if (instruction.opcode == "V_FMA_F64") {
      std::uint64_t src0 = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[1], *state, lane_index,
                                      &src0, error_message)) {
        return false;
      }
      std::uint64_t src1 = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[2], *state, lane_index,
                                      &src1, error_message)) {
        return false;
      }
      std::uint64_t src2 = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[3], *state, lane_index,
                                      &src2, error_message)) {
        return false;
      }
      const std::uint64_t result = EvaluateVectorFloatTernaryF64(src0, src1, src2);
      if (!WriteVectorPairOperandValue(instruction.operands[0], lane_index, result,
                                       state, error_message)) {
        return false;
      }
      continue;
    }

    const std::uint32_t src0 =
        ReadVectorOperand(instruction.operands[1], *state, lane_index, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint32_t src1 =
        ReadVectorOperand(instruction.operands[2], *state, lane_index, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint32_t src2 =
        ReadVectorOperand(instruction.operands[3], *state, lane_index, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }

    std::uint32_t result = 0;
    if (instruction.opcode == "V_ADD3_U32") {
      result = static_cast<std::uint32_t>(static_cast<std::uint64_t>(src0) +
                                          static_cast<std::uint64_t>(src1) +
                                          static_cast<std::uint64_t>(src2));
    } else if (instruction.opcode == "V_FMA_F32") {
      result = EvaluateVectorFloatTernaryF32(src0, src1, src2);
    } else if (instruction.opcode == "V_LSHL_ADD_U32") {
      result = (src0 << (src1 & 31u)) + src2;
    } else if (instruction.opcode == "V_ADD_LSHL_U32") {
      result = static_cast<std::uint32_t>(
          (static_cast<std::uint64_t>(src0) + static_cast<std::uint64_t>(src1))
          << (src2 & 31u));
    } else if (instruction.opcode == "V_LSHL_OR_B32") {
      result = (src0 << (src1 & 31u)) | src2;
    } else if (instruction.opcode == "V_AND_OR_B32") {
      result = (src0 & src1) | src2;
    } else if (instruction.opcode == "V_OR3_B32") {
      result = src0 | src1 | src2;
    } else if (instruction.opcode == "V_XAD_U32") {
      result = (src0 ^ src1) + src2;
    } else if (instruction.opcode == "V_LERP_U8") {
      result = LerpU8(src0, src1, src2);
    } else if (instruction.opcode == "V_PERM_B32") {
      result = PermB32(src0, src1, src2);
    } else if (instruction.opcode == "V_BFE_U32") {
      const std::uint32_t packed_field =
          (src1 & 31u) | ((src2 & 0x7fu) << 16);
      result = ExtractUnsignedBitfield32(src0, packed_field);
    } else if (instruction.opcode == "V_BFE_I32") {
      const std::uint32_t packed_field =
          (src1 & 31u) | ((src2 & 0x7fu) << 16);
      result = ExtractSignedBitfield32(src0, packed_field);
    } else if (instruction.opcode == "V_BFI_B32") {
      result = (src1 & src0) | (src2 & ~src0);
    } else if (instruction.opcode == "V_ALIGNBIT_B32") {
      const std::uint32_t shift = src2 & 31u;
      const std::uint64_t value =
          (static_cast<std::uint64_t>(src0) << 32) | src1;
      result = static_cast<std::uint32_t>(value >> shift);
    } else if (instruction.opcode == "V_ALIGNBYTE_B32") {
      const std::uint32_t shift_bytes = src2 & 7u;
      const std::uint64_t value =
          (static_cast<std::uint64_t>(src0) << 32) | src1;
      result = static_cast<std::uint32_t>(value >> (shift_bytes * 8u));
    } else if (instruction.opcode == "V_MIN3_I32") {
      result = Min3Signed32(src0, src1, src2);
    } else if (instruction.opcode == "V_MIN3_U32") {
      result = Min3Unsigned32(src0, src1, src2);
    } else if (instruction.opcode == "V_MAX3_I32") {
      result = Max3Signed32(src0, src1, src2);
    } else if (instruction.opcode == "V_MAX3_U32") {
      result = Max3Unsigned32(src0, src1, src2);
    } else if (instruction.opcode == "V_MED3_I32") {
      result = Med3Signed32(src0, src1, src2);
    } else if (instruction.opcode == "V_MED3_U32") {
      result = Med3Unsigned32(src0, src1, src2);
    } else if (instruction.opcode == "V_SAD_U8") {
      result = SumAbsoluteDifferencesU8(src0, src1) + src2;
    } else if (instruction.opcode == "V_SAD_HI_U8") {
      result = (SumAbsoluteDifferencesU8(src0, src1) << 16u) + src2;
    } else if (instruction.opcode == "V_SAD_U16") {
      result = SumAbsoluteDifferencesU16(src0, src1) + src2;
    } else if (instruction.opcode == "V_SAD_U32") {
      const std::uint32_t difference =
          src0 >= src1 ? (src0 - src1) : (src1 - src0);
      result = difference + src2;
    } else if (instruction.opcode == "V_MAD_I32_I24") {
      result = MadI32I24(src0, src1, src2);
    } else if (instruction.opcode == "V_MAD_U32_U24") {
      result = MadU32U24(src0, src1, src2);
    } else {
      if (error_message != nullptr) {
        *error_message = "unsupported vector ternary opcode";
      }
      return false;
    }

    if (!WriteVectorOperand(instruction.operands[0], lane_index, result, state,
                            error_message)) {
      return false;
    }
  }
  return true;
}

bool Gfx950Interpreter::ExecuteVectorTernary(
    const CompiledInstruction& instruction,
    WaveExecutionState* state,
    std::string* error_message) const {
  if ((instruction.opcode == CompiledOpcode::kVMadU64U32 ||
       instruction.opcode == CompiledOpcode::kVMadI64I32)) {
    if (!ValidateOperandCount(instruction, 5, error_message)) {
      return false;
    }
  } else if (!ValidateOperandCount(instruction, 4, error_message)) {
    return false;
  }

  for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
       ++lane_index) {
    if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
      continue;
    }

    if (instruction.opcode == CompiledOpcode::kVLshlAddU64) {
      std::uint64_t src0 = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[1], *state, lane_index,
                                      &src0, error_message)) {
        return false;
      }
      const std::uint32_t shift_count =
          ReadVectorOperand(instruction.operands[2], *state, lane_index,
                            error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      std::uint64_t src2 = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[3], *state, lane_index,
                                      &src2, error_message)) {
        return false;
      }
      const std::uint64_t result = (src0 << (shift_count & 63u)) + src2;
      if (!WriteVectorPairOperandValue(instruction.operands[0], lane_index, result,
                                       state, error_message)) {
        return false;
      }
      continue;
    }

    if (instruction.opcode == CompiledOpcode::kVMadU64U32 ||
        instruction.opcode == CompiledOpcode::kVMadI64I32) {
      const std::uint32_t src0 =
          ReadVectorOperand(instruction.operands[2], *state, lane_index,
                            error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      const std::uint32_t src1 =
          ReadVectorOperand(instruction.operands[3], *state, lane_index,
                            error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      std::uint64_t src2 = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[4], *state, lane_index,
                                      &src2, error_message)) {
        return false;
      }

      std::uint64_t result = 0;
      bool carry_or_overflow = false;
      if (instruction.opcode == CompiledOpcode::kVMadU64U32) {
        const unsigned __int128 full =
            static_cast<unsigned __int128>(src0) *
                static_cast<unsigned __int128>(src1) +
            static_cast<unsigned __int128>(src2);
        result = static_cast<std::uint64_t>(full);
        carry_or_overflow = (full >> 64) != 0;
      } else {
        const __int128 full =
            static_cast<__int128>(static_cast<std::int64_t>(
                BitCast<std::int32_t>(src0))) *
                static_cast<__int128>(static_cast<std::int64_t>(
                    BitCast<std::int32_t>(src1))) +
            static_cast<__int128>(BitCast<std::int64_t>(src2));
        result = static_cast<std::uint64_t>(full);
        carry_or_overflow =
            full > static_cast<__int128>(std::numeric_limits<std::int64_t>::max()) ||
            full < static_cast<__int128>(std::numeric_limits<std::int64_t>::min());
      }

      if (!WriteVectorPairOperandValue(instruction.operands[0], lane_index, result,
                                       state, error_message)) {
        return false;
      }
      if (instruction.operands[1].kind == OperandKind::kSgpr &&
          instruction.operands[1].index + 1 < state->sgprs.size()) {
        std::uint64_t mask =
            ReadScalarPairOperand(instruction.operands[1], *state, error_message);
        if (error_message != nullptr && !error_message->empty()) {
          return false;
        }
        if (carry_or_overflow) {
          mask |= (1ULL << lane_index);
        } else {
          mask &= ~(1ULL << lane_index);
        }
        if (!WriteScalarPairOperand(instruction.operands[1], mask, state,
                                    error_message)) {
          return false;
        }
        if (instruction.operands[1].index == kVccPairSgprIndex) {
          state->vcc_mask = mask;
        }
      }
      continue;
    }

    if (instruction.opcode == CompiledOpcode::kVFmaF64) {
      std::uint64_t src0 = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[1], *state, lane_index,
                                      &src0, error_message)) {
        return false;
      }
      std::uint64_t src1 = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[2], *state, lane_index,
                                      &src1, error_message)) {
        return false;
      }
      std::uint64_t src2 = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[3], *state, lane_index,
                                      &src2, error_message)) {
        return false;
      }
      const std::uint64_t result = EvaluateVectorFloatTernaryF64(src0, src1, src2);
      if (!WriteVectorPairOperandValue(instruction.operands[0], lane_index, result,
                                       state, error_message)) {
        return false;
      }
      continue;
    }

    const std::uint32_t src0 =
        ReadVectorOperand(instruction.operands[1], *state, lane_index, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint32_t src1 =
        ReadVectorOperand(instruction.operands[2], *state, lane_index, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    const std::uint32_t src2 =
        ReadVectorOperand(instruction.operands[3], *state, lane_index, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }

    std::uint32_t result = 0;
    switch (instruction.opcode) {
      case CompiledOpcode::kVAdd3U32:
        result = static_cast<std::uint32_t>(static_cast<std::uint64_t>(src0) +
                                            static_cast<std::uint64_t>(src1) +
                                            static_cast<std::uint64_t>(src2));
        break;
      case CompiledOpcode::kVFmaF32:
        result = EvaluateVectorFloatTernaryF32(src0, src1, src2);
        break;
      case CompiledOpcode::kVLshlAddU32:
        result = (src0 << (src1 & 31u)) + src2;
        break;
      case CompiledOpcode::kVAddLshlU32:
        result = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(src0) +
             static_cast<std::uint64_t>(src1))
            << (src2 & 31u));
        break;
      case CompiledOpcode::kVLshlOrB32:
        result = (src0 << (src1 & 31u)) | src2;
        break;
      case CompiledOpcode::kVAndOrB32:
        result = (src0 & src1) | src2;
        break;
      case CompiledOpcode::kVOr3B32:
        result = src0 | src1 | src2;
        break;
      case CompiledOpcode::kVXadU32:
        result = (src0 ^ src1) + src2;
        break;
      case CompiledOpcode::kVLerpU8:
        result = LerpU8(src0, src1, src2);
        break;
      case CompiledOpcode::kVPermB32:
        result = PermB32(src0, src1, src2);
        break;
      case CompiledOpcode::kVBfeU32: {
        const std::uint32_t packed_field =
            (src1 & 31u) | ((src2 & 0x7fu) << 16);
        result = ExtractUnsignedBitfield32(src0, packed_field);
        break;
      }
      case CompiledOpcode::kVBfeI32: {
        const std::uint32_t packed_field =
            (src1 & 31u) | ((src2 & 0x7fu) << 16);
        result = ExtractSignedBitfield32(src0, packed_field);
        break;
      }
      case CompiledOpcode::kVBfiB32:
        result = (src1 & src0) | (src2 & ~src0);
        break;
      case CompiledOpcode::kVAlignbitB32: {
        const std::uint32_t shift = src2 & 31u;
        const std::uint64_t value =
            (static_cast<std::uint64_t>(src0) << 32) | src1;
        result = static_cast<std::uint32_t>(value >> shift);
        break;
      }
      case CompiledOpcode::kVAlignbyteB32: {
        const std::uint32_t shift_bytes = src2 & 7u;
        const std::uint64_t value =
            (static_cast<std::uint64_t>(src0) << 32) | src1;
        result = static_cast<std::uint32_t>(value >> (shift_bytes * 8u));
        break;
      }
      case CompiledOpcode::kVMin3I32:
        result = Min3Signed32(src0, src1, src2);
        break;
      case CompiledOpcode::kVMin3U32:
        result = Min3Unsigned32(src0, src1, src2);
        break;
      case CompiledOpcode::kVMax3I32:
        result = Max3Signed32(src0, src1, src2);
        break;
      case CompiledOpcode::kVMax3U32:
        result = Max3Unsigned32(src0, src1, src2);
        break;
      case CompiledOpcode::kVMed3I32:
        result = Med3Signed32(src0, src1, src2);
        break;
      case CompiledOpcode::kVMed3U32:
        result = Med3Unsigned32(src0, src1, src2);
        break;
      case CompiledOpcode::kVSadU8:
        result = SumAbsoluteDifferencesU8(src0, src1) + src2;
        break;
      case CompiledOpcode::kVSadHiU8:
        result = (SumAbsoluteDifferencesU8(src0, src1) << 16u) + src2;
        break;
      case CompiledOpcode::kVSadU16:
        result = SumAbsoluteDifferencesU16(src0, src1) + src2;
        break;
      case CompiledOpcode::kVSadU32: {
        const std::uint32_t difference =
            src0 >= src1 ? (src0 - src1) : (src1 - src0);
        result = difference + src2;
        break;
      }
      case CompiledOpcode::kVMadI32I24:
        result = MadI32I24(src0, src1, src2);
        break;
      case CompiledOpcode::kVMadU32U24:
        result = MadU32U24(src0, src1, src2);
        break;
      default:
        if (error_message != nullptr) {
          *error_message = "unsupported compiled vector ternary opcode";
        }
        return false;
    }

    if (!WriteVectorOperand(instruction.operands[0], lane_index, result, state,
                            error_message)) {
      return false;
    }
  }
  return true;
}

bool Gfx950Interpreter::ExecuteVectorCompare(const DecodedInstruction& instruction,
                                             WaveExecutionState* state,
                                             std::string* error_message) const {
  if (!ValidateOperandCount(instruction, 3, error_message)) {
    return false;
  }
  if (instruction.operands[0].kind != OperandKind::kSgpr) {
    if (error_message != nullptr) {
      *error_message = "vector compare requires scalar destination pair";
    }
    return false;
  }

  const bool writes_exec = IsVectorCmpxOpcode(instruction.opcode);
  std::uint64_t mask = writes_exec ? 0ULL : state->vcc_mask;
  for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
       ++lane_index) {
    if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
      continue;
    }

    bool compare_result = false;
    if (IsVectorCompareClassOpcode(instruction.opcode)) {
      if (IsVectorCompareClass64Opcode(instruction.opcode)) {
        std::uint64_t lhs = 0;
        if (!ReadVectorPairOperandValue(instruction.operands[1], *state, lane_index,
                                        &lhs, error_message)) {
          return false;
        }
        const std::uint32_t class_mask =
            ReadVectorOperand(instruction.operands[2], *state, lane_index,
                              error_message);
        if (error_message != nullptr && !error_message->empty()) {
          return false;
        }
        if (!EvaluateVectorCompareClassOpcode(instruction.opcode, lhs, class_mask,
                                              &compare_result)) {
          if (error_message != nullptr) {
            *error_message = "unsupported vector compare class opcode";
          }
          return false;
        }
      } else {
        const std::uint32_t lhs =
            ReadVectorOperand(instruction.operands[1], *state, lane_index,
                              error_message);
        if (error_message != nullptr && !error_message->empty()) {
          return false;
        }
        const std::uint32_t class_mask =
            ReadVectorOperand(instruction.operands[2], *state, lane_index,
                              error_message);
        if (error_message != nullptr && !error_message->empty()) {
          return false;
        }
        if (!EvaluateVectorCompareClassOpcode(instruction.opcode, lhs, class_mask,
                                              &compare_result)) {
          if (error_message != nullptr) {
            *error_message = "unsupported vector compare class opcode";
          }
          return false;
        }
      }
    } else if (IsVectorCompare64Opcode(instruction.opcode)) {
      std::uint64_t lhs = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[1], *state, lane_index,
                                      &lhs, error_message)) {
        return false;
      }
      std::uint64_t rhs = 0;
      if (!ReadVectorPairOperandValue(instruction.operands[2], *state, lane_index,
                                      &rhs, error_message)) {
        return false;
      }
      if (!EvaluateVectorCompareOpcode(instruction.opcode, lhs, rhs,
                                       &compare_result)) {
        if (error_message != nullptr) {
          *error_message = "unsupported vector compare opcode";
        }
        return false;
      }
    } else {
      const std::uint32_t lhs =
          ReadVectorOperand(instruction.operands[1], *state, lane_index,
                            error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      const std::uint32_t rhs =
          ReadVectorOperand(instruction.operands[2], *state, lane_index,
                            error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      if (!EvaluateVectorCompareOpcode(instruction.opcode, lhs, rhs,
                                       &compare_result)) {
        if (error_message != nullptr) {
          *error_message = "unsupported vector compare opcode";
        }
        return false;
      }
    }

    const std::uint64_t lane_bit = 1ULL << lane_index;
    if (compare_result) {
      mask |= lane_bit;
    } else {
      mask &= ~lane_bit;
    }
  }

  state->vcc_mask = mask;
  if (writes_exec) {
    state->exec_mask = mask;
  }
  return WriteScalarPairOperand(instruction.operands[0], mask, state, error_message);
}

bool Gfx950Interpreter::ExecuteVectorCompare(
    const CompiledInstruction& instruction,
    WaveExecutionState* state,
    std::string* error_message) const {
  if (!ValidateOperandCount(instruction, 3, error_message)) {
    return false;
  }
  if (instruction.operands[0].kind != OperandKind::kSgpr) {
    if (error_message != nullptr) {
      *error_message = "vector compare requires scalar destination pair";
    }
    return false;
  }

  const bool writes_exec = IsVectorCmpxOpcode(instruction.opcode);
  std::uint64_t mask = writes_exec ? 0ULL : state->vcc_mask;
  for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
       ++lane_index) {
    if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
      continue;
    }

    bool compare_result = false;
    if (IsVectorCompareClassOpcode(instruction.opcode)) {
      if (IsVectorCompareClass64Opcode(instruction.opcode)) {
        std::uint64_t lhs = 0;
        if (!ReadVectorPairOperandValue(instruction.operands[1], *state, lane_index,
                                        &lhs, error_message)) {
          return false;
        }
        const std::uint32_t class_mask =
            ReadVectorOperand(instruction.operands[2], *state, lane_index,
                              error_message);
        if (error_message != nullptr && !error_message->empty()) {
          return false;
        }
        if (!EvaluateVectorCompareClassOpcode(instruction.opcode, lhs, class_mask,
                                              &compare_result)) {
          if (error_message != nullptr) {
            *error_message = "unsupported compiled vector compare class opcode";
          }
          return false;
        }
      } else {
        const std::uint32_t lhs =
            ReadVectorOperand(instruction.operands[1], *state, lane_index,
                              error_message);
        if (error_message != nullptr && !error_message->empty()) {
          return false;
        }
        const std::uint32_t class_mask =
            ReadVectorOperand(instruction.operands[2], *state, lane_index,
                              error_message);
        if (error_message != nullptr && !error_message->empty()) {
          return false;
        }
        if (!EvaluateVectorCompareClassOpcode(instruction.opcode, lhs, class_mask,
                                              &compare_result)) {
          if (error_message != nullptr) {
            *error_message = "unsupported compiled vector compare class opcode";
          }
          return false;
        }
      }
    } else {
      switch (instruction.opcode) {
        case CompiledOpcode::kVCmpFF64:
        case CompiledOpcode::kVCmpLtF64:
        case CompiledOpcode::kVCmpEqF64:
        case CompiledOpcode::kVCmpLeF64:
        case CompiledOpcode::kVCmpGtF64:
        case CompiledOpcode::kVCmpLgF64:
        case CompiledOpcode::kVCmpGeF64:
        case CompiledOpcode::kVCmpOF64:
        case CompiledOpcode::kVCmpUF64:
        case CompiledOpcode::kVCmpNgeF64:
        case CompiledOpcode::kVCmpNlgF64:
        case CompiledOpcode::kVCmpNgtF64:
        case CompiledOpcode::kVCmpNleF64:
        case CompiledOpcode::kVCmpNeqF64:
        case CompiledOpcode::kVCmpNltF64:
        case CompiledOpcode::kVCmpTruF64:
        case CompiledOpcode::kVCmpFI64:
        case CompiledOpcode::kVCmpLtI64:
        case CompiledOpcode::kVCmpEqI64:
        case CompiledOpcode::kVCmpLeI64:
        case CompiledOpcode::kVCmpGtI64:
        case CompiledOpcode::kVCmpNeI64:
        case CompiledOpcode::kVCmpGeI64:
        case CompiledOpcode::kVCmpTI64:
        case CompiledOpcode::kVCmpFU64:
        case CompiledOpcode::kVCmpLtU64:
        case CompiledOpcode::kVCmpEqU64:
        case CompiledOpcode::kVCmpLeU64:
        case CompiledOpcode::kVCmpGtU64:
        case CompiledOpcode::kVCmpNeU64:
        case CompiledOpcode::kVCmpGeU64:
        case CompiledOpcode::kVCmpxFF64:
        case CompiledOpcode::kVCmpxLtF64:
        case CompiledOpcode::kVCmpxEqF64:
        case CompiledOpcode::kVCmpxLeF64:
        case CompiledOpcode::kVCmpxGtF64:
        case CompiledOpcode::kVCmpxLgF64:
        case CompiledOpcode::kVCmpxGeF64:
        case CompiledOpcode::kVCmpxOF64:
        case CompiledOpcode::kVCmpxUF64:
        case CompiledOpcode::kVCmpxNgeF64:
        case CompiledOpcode::kVCmpxNlgF64:
        case CompiledOpcode::kVCmpxNgtF64:
        case CompiledOpcode::kVCmpxNleF64:
        case CompiledOpcode::kVCmpxNeqF64:
        case CompiledOpcode::kVCmpxNltF64:
        case CompiledOpcode::kVCmpxTruF64:
        case CompiledOpcode::kVCmpxFI64:
        case CompiledOpcode::kVCmpxLtI64:
        case CompiledOpcode::kVCmpxEqI64:
        case CompiledOpcode::kVCmpxLeI64:
        case CompiledOpcode::kVCmpxGtI64:
        case CompiledOpcode::kVCmpxNeI64:
        case CompiledOpcode::kVCmpxGeI64:
        case CompiledOpcode::kVCmpxTI64:
        case CompiledOpcode::kVCmpxFU64:
        case CompiledOpcode::kVCmpxLtU64:
        case CompiledOpcode::kVCmpxEqU64:
        case CompiledOpcode::kVCmpxLeU64:
        case CompiledOpcode::kVCmpxGtU64:
        case CompiledOpcode::kVCmpxNeU64:
        case CompiledOpcode::kVCmpxGeU64:
        case CompiledOpcode::kVCmpTU64:
        case CompiledOpcode::kVCmpxTU64: {
          std::uint64_t lhs = 0;
          if (!ReadVectorPairOperandValue(instruction.operands[1], *state, lane_index,
                                          &lhs, error_message)) {
            return false;
          }
          std::uint64_t rhs = 0;
          if (!ReadVectorPairOperandValue(instruction.operands[2], *state, lane_index,
                                          &rhs, error_message)) {
            return false;
          }
          if (!EvaluateVectorCompareOpcode(instruction.opcode, lhs, rhs,
                                           &compare_result)) {
            if (error_message != nullptr) {
              *error_message = "unsupported compiled vector compare opcode";
            }
            return false;
          }
          break;
        }
        default: {
          const std::uint32_t lhs =
              ReadVectorOperand(instruction.operands[1], *state, lane_index,
                                error_message);
          if (error_message != nullptr && !error_message->empty()) {
            return false;
          }
          const std::uint32_t rhs =
              ReadVectorOperand(instruction.operands[2], *state, lane_index,
                                error_message);
          if (error_message != nullptr && !error_message->empty()) {
            return false;
          }
          if (!EvaluateVectorCompareOpcode(instruction.opcode, lhs, rhs,
                                           &compare_result)) {
            if (error_message != nullptr) {
              *error_message = "unsupported compiled vector compare opcode";
            }
            return false;
          }
          break;
        }
      }
    }

    const std::uint64_t lane_bit = 1ULL << lane_index;
    if (compare_result) {
      mask |= lane_bit;
    } else {
      mask &= ~lane_bit;
    }
  }

  state->vcc_mask = mask;
  if (writes_exec) {
    state->exec_mask = mask;
  }
  return WriteScalarPairOperand(instruction.operands[0], mask, state, error_message);
}

bool Gfx950Interpreter::ExecuteVectorMemory(const DecodedInstruction& instruction,
                                            WaveExecutionState* state,
                                            ExecutionMemory* memory,
                                            std::string* error_message) const {
  const bool is_global = IsGlobalVectorMemoryOpcode(instruction.opcode);
  const bool is_load = IsVectorMemoryLoadOpcode(instruction.opcode);
  const std::uint8_t register_dword_count =
      GetVectorMemoryRegisterDwordCount(instruction.opcode);
  const std::uint8_t element_size_bytes =
      GetVectorMemoryElementSizeBytes(instruction.opcode);
  const std::uint8_t expected_operands = is_global ? 4 : 3;
  if (!ValidateOperandCount(instruction, expected_operands, error_message)) {
    return false;
  }
  if (memory == nullptr) {
    if (error_message != nullptr) {
      *error_message = "vector memory instruction requires execution memory";
    }
    return false;
  }

  const InstructionOperand& address_operand =
      instruction.operands[is_load ? 1 : 0];
  const InstructionOperand& offset_operand =
      instruction.operands[expected_operands - 1];
  const InstructionOperand& vector_data_operand =
      instruction.operands[is_load ? 0 : 1];
  if (vector_data_operand.kind != OperandKind::kVgpr) {
    if (error_message != nullptr) {
      *error_message = "vector memory data operand must be a VGPR";
    }
    return false;
  }
  if (vector_data_operand.index + register_dword_count - 1 >= state->vgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "vector memory data register range out of bounds";
    }
    return false;
  }
  if (offset_operand.kind != OperandKind::kImm32) {
    if (error_message != nullptr) {
      *error_message = "vector memory offset must be an immediate";
    }
    return false;
  }

  std::uint64_t scalar_base = 0;
  const InstructionOperand* scalar_base_operand =
      is_global ? &instruction.operands[2] : nullptr;
  const std::int32_t signed_offset =
      static_cast<std::int32_t>(offset_operand.imm32);

  for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
       ++lane_index) {
    if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
      continue;
    }

    std::uint64_t address = 0;
    if (!ResolveVectorMemoryAddress(address_operand, scalar_base_operand,
                                    signed_offset, *state, lane_index, &address,
                                    error_message)) {
      return false;
    }

    for (std::uint8_t element_index = 0; element_index < register_dword_count;
         ++element_index) {
      const std::uint64_t byte_offset =
          static_cast<std::uint64_t>(element_index) * element_size_bytes;
      if (address > std::numeric_limits<std::uint64_t>::max() - byte_offset) {
        if (error_message != nullptr) {
          *error_message = "vector memory element address overflow";
        }
        return false;
      }
      const std::uint64_t element_address = address + byte_offset;
      const InstructionOperand element_operand = InstructionOperand::Vgpr(
          static_cast<std::uint16_t>(vector_data_operand.index + element_index));

      if (is_load) {
        std::uint32_t value = 0;
        if (element_size_bytes == 1) {
          std::uint8_t raw_value = 0;
          if (!ReadMemoryU8(memory, element_address, &raw_value, error_message)) {
            return false;
          }
          if (IsSignedVectorMemoryLoadOpcode(instruction.opcode)) {
            value = static_cast<std::uint32_t>(
                static_cast<std::int32_t>(static_cast<std::int8_t>(raw_value)));
          } else {
            value = raw_value;
          }
        } else if (element_size_bytes == 2) {
          std::uint16_t raw_value = 0;
          if (!ReadMemoryU16(memory, element_address, &raw_value, error_message)) {
            return false;
          }
          if (IsSignedVectorMemoryLoadOpcode(instruction.opcode)) {
            value = static_cast<std::uint32_t>(
                static_cast<std::int32_t>(static_cast<std::int16_t>(raw_value)));
          } else {
            value = raw_value;
          }
        } else if (!ReadMemoryU32(memory, element_address, &value, error_message)) {
          return false;
        }

        if (!WriteVectorOperand(element_operand, lane_index, value, state,
                                error_message)) {
          return false;
        }
        continue;
      }

      const std::uint32_t value =
          ReadVectorOperand(element_operand, *state, lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      if ((element_size_bytes == 1 &&
           !WriteMemoryU8(memory, element_address,
                          static_cast<std::uint8_t>(value), error_message)) ||
          (element_size_bytes == 2 &&
           !WriteMemoryU16(memory, element_address,
                           static_cast<std::uint16_t>(value), error_message)) ||
          (element_size_bytes == 4 &&
           !WriteMemoryU32(memory, element_address, value, error_message))) {
        return false;
      }
    }
  }
  return true;
}

bool Gfx950Interpreter::ExecuteVectorMemory(const CompiledInstruction& instruction,
                                            WaveExecutionState* state,
                                            ExecutionMemory* memory,
                                            std::string* error_message) const {
  const bool is_global = instruction.IsGlobal();
  const bool is_load = instruction.IsLoad();
  const std::uint8_t register_dword_count = instruction.register_dword_count;
  const std::uint8_t element_size_bytes = instruction.element_size_bytes;
  const std::uint8_t expected_operands = is_global ? 4 : 3;
  if (!ValidateOperandCount(instruction, expected_operands, error_message)) {
    return false;
  }
  if (memory == nullptr) {
    if (error_message != nullptr) {
      *error_message = "vector memory instruction requires execution memory";
    }
    return false;
  }

  const InstructionOperand& address_operand =
      instruction.operands[is_load ? 1 : 0];
  const InstructionOperand& offset_operand =
      instruction.operands[expected_operands - 1];
  const InstructionOperand& vector_data_operand =
      instruction.operands[is_load ? 0 : 1];
  if (address_operand.kind != OperandKind::kVgpr) {
    if (error_message != nullptr) {
      *error_message = "vector memory address operand must be a VGPR pair";
    }
    return false;
  }
  if (vector_data_operand.kind != OperandKind::kVgpr) {
    if (error_message != nullptr) {
      *error_message = "vector memory data operand must be a VGPR";
    }
    return false;
  }
  if (address_operand.index + 1 >= state->vgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "vector memory address register pair out of bounds";
    }
    return false;
  }
  if (vector_data_operand.index + register_dword_count - 1 >= state->vgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "vector memory data register range out of bounds";
    }
    return false;
  }
  if (offset_operand.kind != OperandKind::kImm32) {
    if (error_message != nullptr) {
      *error_message = "vector memory offset must be an immediate";
    }
    return false;
  }

  const InstructionOperand* scalar_base_operand =
      is_global ? &instruction.operands[2] : nullptr;
  std::uint64_t scalar_base = 0;
  if (scalar_base_operand != nullptr) {
    if (scalar_base_operand->kind != OperandKind::kSgpr) {
      if (error_message != nullptr) {
        *error_message = "global vector memory scalar base must be an SGPR pair";
      }
      return false;
    }
    if (scalar_base_operand->index + 1 >= state->sgprs.size()) {
      if (error_message != nullptr) {
        *error_message = "global vector memory scalar base out of bounds";
      }
      return false;
    }
    scalar_base =
        ComposeU64(state->sgprs[scalar_base_operand->index],
                   state->sgprs[scalar_base_operand->index + 1]);
  }
  const std::int32_t signed_offset =
      static_cast<std::int32_t>(offset_operand.imm32);
  const std::uint16_t address_reg = address_operand.index;
  const std::uint16_t vector_data_reg = vector_data_operand.index;

  for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
       ++lane_index) {
    if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
      continue;
    }

    std::uint64_t address =
        ComposeU64(state->vgprs[address_reg][lane_index],
                   state->vgprs[address_reg + 1][lane_index]);
    if (scalar_base_operand != nullptr) {
      if (address > std::numeric_limits<std::uint64_t>::max() - scalar_base) {
        if (error_message != nullptr) {
          *error_message = "global address overflow";
        }
        return false;
      }
      address += scalar_base;
    }
    if (signed_offset < 0) {
      const std::uint64_t magnitude =
          static_cast<std::uint64_t>(-signed_offset);
      if (address < magnitude) {
        if (error_message != nullptr) {
          *error_message = "vector memory address underflow";
        }
        return false;
      }
      address -= magnitude;
    } else {
      const std::uint64_t magnitude =
          static_cast<std::uint64_t>(signed_offset);
      if (address > std::numeric_limits<std::uint64_t>::max() - magnitude) {
        if (error_message != nullptr) {
          *error_message = "vector memory address overflow";
        }
        return false;
      }
      address += magnitude;
    }

    if (element_size_bytes == sizeof(std::uint32_t) && register_dword_count > 1) {
      const std::uint64_t transfer_size_bytes =
          static_cast<std::uint64_t>(register_dword_count) * sizeof(std::uint32_t);
      if (address > std::numeric_limits<std::uint64_t>::max() -
                        (transfer_size_bytes - 1)) {
        if (error_message != nullptr) {
          *error_message = "vector memory transfer address overflow";
        }
        return false;
      }

      std::array<std::uint32_t, 4> transfer_values{};
      if (is_load) {
        if (!memory->Load(
                address,
                std::as_writable_bytes(std::span<std::uint32_t>(
                    transfer_values.data(), register_dword_count)))) {
          if (error_message != nullptr) {
            *error_message = "vector memory transfer load failed";
          }
          return false;
        }
        for (std::uint8_t element_index = 0; element_index < register_dword_count;
             ++element_index) {
          state->vgprs[static_cast<std::uint16_t>(vector_data_reg + element_index)]
                      [lane_index] = transfer_values[element_index];
        }
        continue;
      }

      for (std::uint8_t element_index = 0; element_index < register_dword_count;
           ++element_index) {
        transfer_values[element_index] =
            state->vgprs[static_cast<std::uint16_t>(vector_data_reg + element_index)]
                        [lane_index];
      }
      if (!memory->Store(
              address,
              std::as_bytes(std::span<const std::uint32_t>(
                  transfer_values.data(), register_dword_count)))) {
        if (error_message != nullptr) {
          *error_message = "vector memory transfer store failed";
        }
        return false;
      }
      continue;
    }

    for (std::uint8_t element_index = 0; element_index < register_dword_count;
         ++element_index) {
      const std::uint64_t byte_offset =
          static_cast<std::uint64_t>(element_index) * element_size_bytes;
      if (address > std::numeric_limits<std::uint64_t>::max() - byte_offset) {
        if (error_message != nullptr) {
          *error_message = "vector memory element address overflow";
        }
        return false;
      }
      const std::uint64_t element_address = address + byte_offset;
      const std::uint16_t element_reg =
          static_cast<std::uint16_t>(vector_data_reg + element_index);

      if (is_load) {
        std::uint32_t value = 0;
        if (element_size_bytes == 1) {
          std::uint8_t raw_value = 0;
          if (!ReadMemoryU8(memory, element_address, &raw_value, error_message)) {
            return false;
          }
          if (instruction.IsSignedLoad()) {
            value = static_cast<std::uint32_t>(
                static_cast<std::int32_t>(static_cast<std::int8_t>(raw_value)));
          } else {
            value = raw_value;
          }
        } else if (element_size_bytes == 2) {
          std::uint16_t raw_value = 0;
          if (!ReadMemoryU16(memory, element_address, &raw_value, error_message)) {
            return false;
          }
          if (instruction.IsSignedLoad()) {
            value = static_cast<std::uint32_t>(
                static_cast<std::int32_t>(static_cast<std::int16_t>(raw_value)));
          } else {
            value = raw_value;
          }
        } else if (!ReadMemoryU32(memory, element_address, &value, error_message)) {
          return false;
        }

        state->vgprs[element_reg][lane_index] = value;
        continue;
      }

      const std::uint32_t value = state->vgprs[element_reg][lane_index];
      if ((element_size_bytes == 1 &&
           !WriteMemoryU8(memory, element_address,
                          static_cast<std::uint8_t>(value), error_message)) ||
          (element_size_bytes == 2 &&
           !WriteMemoryU16(memory, element_address,
                           static_cast<std::uint16_t>(value), error_message)) ||
          (element_size_bytes == 4 &&
           !WriteMemoryU32(memory, element_address, value, error_message))) {
        return false;
      }
    }
  }
  return true;
}

bool Gfx950Interpreter::ExecuteDsMemory(const DecodedInstruction& instruction,
                                        WaveExecutionState* state,
                                        const WorkgroupExecutionContext* workgroup,
                                        std::string* error_message) const {
  const bool is_pair_write = IsDsPairWriteOpcode(instruction.opcode);
  const bool is_pair_read = IsDsPairReadOpcode(instruction.opcode);
  const bool is_narrow_read = IsDsNarrowReadOpcode(instruction.opcode);
  const bool is_dual_data = IsDsDualDataOpcode(instruction.opcode);
  const bool is_dual_data_return =
      IsDsDualDataReturnOpcode(instruction.opcode);
  const bool is_return =
      IsDsReturnOpcode(instruction.opcode) || is_dual_data_return;
  const bool is_read =
      instruction.opcode == "DS_READ_B32" || is_pair_read || is_narrow_read;
  const std::uint8_t expected_operands =
      is_pair_write ? 5
                    : (is_pair_read ? 4
                                    : (is_dual_data_return
                                           ? 5
                                           : (is_dual_data ? 4
                                                           : (is_return ? 4
                                                                        : 3))));
  if (!ValidateOperandCount(instruction, expected_operands, error_message)) {
    return false;
  }

  const InstructionOperand* destination_operand = nullptr;
  const InstructionOperand* address_operand = nullptr;
  const InstructionOperand* data_operand = nullptr;
  const InstructionOperand* second_data_operand = nullptr;
  const InstructionOperand* offset0_operand = nullptr;
  const InstructionOperand* offset1_operand = nullptr;
  if (is_pair_write) {
    address_operand = &instruction.operands[0];
    data_operand = &instruction.operands[1];
    second_data_operand = &instruction.operands[2];
    offset0_operand = &instruction.operands[3];
    offset1_operand = &instruction.operands[4];
  } else if (is_pair_read) {
    destination_operand = &instruction.operands[0];
    address_operand = &instruction.operands[1];
    offset0_operand = &instruction.operands[2];
    offset1_operand = &instruction.operands[3];
  } else if (is_dual_data_return) {
    destination_operand = &instruction.operands[0];
    address_operand = &instruction.operands[1];
    data_operand = &instruction.operands[2];
    second_data_operand = &instruction.operands[3];
    offset0_operand = &instruction.operands[4];
  } else if (is_return) {
    destination_operand = &instruction.operands[0];
    address_operand = &instruction.operands[1];
    data_operand = &instruction.operands[2];
    offset0_operand = &instruction.operands[3];
  } else if (is_read) {
    destination_operand = &instruction.operands[0];
    address_operand = &instruction.operands[1];
    offset0_operand = &instruction.operands[2];
  } else if (is_dual_data) {
    address_operand = &instruction.operands[0];
    data_operand = &instruction.operands[1];
    second_data_operand = &instruction.operands[2];
    offset0_operand = &instruction.operands[3];
  } else {
    address_operand = &instruction.operands[0];
    data_operand = &instruction.operands[1];
    offset0_operand = &instruction.operands[2];
  }

  if (address_operand->kind != OperandKind::kVgpr ||
      (data_operand != nullptr && data_operand->kind != OperandKind::kVgpr) ||
      (second_data_operand != nullptr &&
       second_data_operand->kind != OperandKind::kVgpr)) {
    if (error_message != nullptr) {
      *error_message = "ds operands must use VGPRs";
    }
    return false;
  }
  if (destination_operand != nullptr &&
      destination_operand->kind != OperandKind::kVgpr) {
    if (error_message != nullptr) {
      *error_message = "ds destination must use a VGPR";
    }
    return false;
  }
  if (offset0_operand->kind != OperandKind::kImm32 ||
      (offset1_operand != nullptr && offset1_operand->kind != OperandKind::kImm32)) {
    if (error_message != nullptr) {
      *error_message = "ds offset must be an immediate";
    }
    return false;
  }
  if (is_pair_read &&
      destination_operand->index + 1u >= state->vgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "ds read2 destination pair is out of bounds";
    }
    return false;
  }

  const std::size_t access_size = GetDsAccessSize(instruction.opcode);
  const bool sign_extend = IsDsSignedReadOpcode(instruction.opcode);
  const std::uint32_t offset_scale =
      is_pair_write || is_pair_read ? GetDsPairOffsetScale(instruction.opcode) : 1u;
  std::span<std::byte> lds_storage(
      state->lds_bytes.data(), state->lds_bytes.size());
  if (workgroup != nullptr && !workgroup->shared_lds.empty()) {
    lds_storage = workgroup->shared_lds;
  }
  for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
       ++lane_index) {
    if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
      continue;
    }

    const std::uint32_t base_address =
        ReadVectorOperand(*address_operand, *state, lane_index, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    std::uint64_t lds_address0 = 0;
    if (!ComputeDsAddress(base_address, offset0_operand->imm32, offset_scale,
                          &lds_address0, error_message)) {
      return false;
    }
    std::uint64_t lds_address1 = 0;
    if ((is_pair_write || is_pair_read) &&
        !ComputeDsAddress(base_address, offset1_operand->imm32, offset_scale,
                          &lds_address1, error_message)) {
      return false;
    }

    if (is_pair_write) {
      const std::uint32_t data_value0 =
          ReadVectorOperand(*data_operand, *state, lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      const std::uint32_t data_value1 = ReadVectorOperand(
          *second_data_operand, *state, lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      if (!WriteLdsValue(lds_storage, lds_address0, access_size, data_value0,
                         error_message) ||
          !WriteLdsValue(lds_storage, lds_address1, access_size, data_value1,
                         error_message)) {
        return false;
      }
      continue;
    }

    if (is_pair_read) {
      std::uint32_t lds_value0 = 0;
      std::uint32_t lds_value1 = 0;
      if (!ReadLdsValue(lds_storage, lds_address0, access_size, false,
                        &lds_value0, error_message) ||
          !ReadLdsValue(lds_storage, lds_address1, access_size, false,
                        &lds_value1, error_message)) {
        return false;
      }
      if (!WriteVectorOperand(*destination_operand, lane_index, lds_value0, state,
                              error_message)) {
        return false;
      }
      const InstructionOperand second_destination = InstructionOperand::Vgpr(
          static_cast<std::uint16_t>(destination_operand->index + 1u));
      if (!WriteVectorOperand(second_destination, lane_index, lds_value1, state,
                              error_message)) {
        return false;
      }
      continue;
    }

    if (is_read) {
      std::uint32_t lds_value = 0;
      if (!ReadLdsValue(lds_storage, lds_address0, access_size, sign_extend,
                        &lds_value, error_message) ||
          !WriteVectorOperand(*destination_operand, lane_index, lds_value, state,
                              error_message)) {
        return false;
      }
      continue;
    }

    if (instruction.opcode == "DS_WRITE_B32" || instruction.opcode == "DS_WRITE_B8" ||
        instruction.opcode == "DS_WRITE_B16") {
      const std::uint32_t data_value =
          ReadVectorOperand(*data_operand, *state, lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      if (!WriteLdsValue(lds_storage, lds_address0, access_size, data_value,
                         error_message)) {
        return false;
      }
      continue;
    }

    if (is_dual_data || is_dual_data_return) {
      const std::uint32_t data0_value =
          ReadVectorOperand(*data_operand, *state, lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      const std::uint32_t data1_value = ReadVectorOperand(
          *second_data_operand, *state, lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }

      std::uint32_t lds_value = 0;
      if (!ReadLdsValue(lds_storage, lds_address0, access_size, false, &lds_value,
                        error_message)) {
        return false;
      }
      if (is_dual_data_return &&
          !WriteVectorOperand(*destination_operand, lane_index, lds_value, state,
                              error_message)) {
        return false;
      }
      lds_value = EvaluateDsDualDataUpdate(instruction.opcode, lds_value,
                                           data0_value, data1_value,
                                           error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      if (!WriteLdsValue(lds_storage, lds_address0, access_size, lds_value,
                         error_message)) {
        return false;
      }
      continue;
    }

    const std::uint32_t data_value =
        ReadVectorOperand(*data_operand, *state, lane_index, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    std::uint32_t lds_value = 0;
    if (!ReadLdsValue(lds_storage, lds_address0, access_size, false, &lds_value,
                      error_message)) {
      return false;
    }
    if (is_return &&
        !WriteVectorOperand(*destination_operand, lane_index, lds_value, state,
                            error_message)) {
      return false;
    }
    const std::string_view update_opcode = GetDsUpdateOpcode(instruction.opcode);
    lds_value =
        EvaluateDsUpdate(update_opcode, lds_value, data_value, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    if (!WriteLdsValue(lds_storage, lds_address0, access_size, lds_value,
                       error_message)) {
      return false;
    }
  }

  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool Gfx950Interpreter::ExecuteDsMemory(const CompiledInstruction& instruction,
                                        WaveExecutionState* state,
                                        const WorkgroupExecutionContext* workgroup,
                                        std::string* error_message) const {
  const bool is_pair_write = IsDsPairWriteOpcode(instruction.opcode);
  const bool is_pair_read = IsDsPairReadOpcode(instruction.opcode);
  const bool is_narrow_read = IsDsNarrowReadOpcode(instruction.opcode);
  const bool is_dual_data = IsDsDualDataOpcode(instruction.opcode);
  const bool is_dual_data_return =
      IsDsDualDataReturnOpcode(instruction.opcode);
  const bool is_return =
      IsDsReturnOpcode(instruction.opcode) || is_dual_data_return;
  const bool is_read = instruction.opcode == CompiledOpcode::kDsReadB32 ||
                       is_pair_read || is_narrow_read;
  const std::uint8_t expected_operands =
      is_pair_write ? 5
                    : (is_pair_read ? 4
                                    : (is_dual_data_return
                                           ? 5
                                           : (is_dual_data ? 4
                                                           : (is_return ? 4
                                                                        : 3))));
  if (!ValidateOperandCount(instruction, expected_operands, error_message)) {
    return false;
  }

  const InstructionOperand* destination_operand = nullptr;
  const InstructionOperand* address_operand = nullptr;
  const InstructionOperand* data_operand = nullptr;
  const InstructionOperand* second_data_operand = nullptr;
  const InstructionOperand* offset0_operand = nullptr;
  const InstructionOperand* offset1_operand = nullptr;
  if (is_pair_write) {
    address_operand = &instruction.operands[0];
    data_operand = &instruction.operands[1];
    second_data_operand = &instruction.operands[2];
    offset0_operand = &instruction.operands[3];
    offset1_operand = &instruction.operands[4];
  } else if (is_pair_read) {
    destination_operand = &instruction.operands[0];
    address_operand = &instruction.operands[1];
    offset0_operand = &instruction.operands[2];
    offset1_operand = &instruction.operands[3];
  } else if (is_dual_data_return) {
    destination_operand = &instruction.operands[0];
    address_operand = &instruction.operands[1];
    data_operand = &instruction.operands[2];
    second_data_operand = &instruction.operands[3];
    offset0_operand = &instruction.operands[4];
  } else if (is_return) {
    destination_operand = &instruction.operands[0];
    address_operand = &instruction.operands[1];
    data_operand = &instruction.operands[2];
    offset0_operand = &instruction.operands[3];
  } else if (is_read) {
    destination_operand = &instruction.operands[0];
    address_operand = &instruction.operands[1];
    offset0_operand = &instruction.operands[2];
  } else if (is_dual_data) {
    address_operand = &instruction.operands[0];
    data_operand = &instruction.operands[1];
    second_data_operand = &instruction.operands[2];
    offset0_operand = &instruction.operands[3];
  } else {
    address_operand = &instruction.operands[0];
    data_operand = &instruction.operands[1];
    offset0_operand = &instruction.operands[2];
  }

  if (address_operand->kind != OperandKind::kVgpr ||
      (data_operand != nullptr && data_operand->kind != OperandKind::kVgpr) ||
      (second_data_operand != nullptr &&
       second_data_operand->kind != OperandKind::kVgpr)) {
    if (error_message != nullptr) {
      *error_message = "ds operands must use VGPRs";
    }
    return false;
  }
  if (destination_operand != nullptr &&
      destination_operand->kind != OperandKind::kVgpr) {
    if (error_message != nullptr) {
      *error_message = "ds destination must use a VGPR";
    }
    return false;
  }
  if (offset0_operand->kind != OperandKind::kImm32 ||
      (offset1_operand != nullptr && offset1_operand->kind != OperandKind::kImm32)) {
    if (error_message != nullptr) {
      *error_message = "ds offset must be an immediate";
    }
    return false;
  }

  const std::size_t access_size = GetDsAccessSize(instruction.opcode);
  const bool sign_extend = IsDsSignedReadOpcode(instruction.opcode);
  const std::uint32_t offset_scale =
      is_pair_write || is_pair_read ? GetDsPairOffsetScale(instruction.opcode) : 1u;
  if (address_operand->index >= state->vgprs.size() ||
      (data_operand != nullptr && data_operand->index >= state->vgprs.size()) ||
      (second_data_operand != nullptr &&
       second_data_operand->index >= state->vgprs.size())) {
    if (error_message != nullptr) {
      *error_message = "ds register index out of bounds";
    }
    return false;
  }
  if (destination_operand != nullptr &&
      destination_operand->index >= state->vgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "ds destination register index out of bounds";
    }
    return false;
  }
  if (is_pair_read && destination_operand->index + 1u >= state->vgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "ds read2 destination pair is out of bounds";
    }
    return false;
  }
  const std::uint16_t address_reg = address_operand->index;
  const std::uint16_t data_reg = data_operand != nullptr ? data_operand->index : 0;
  const std::uint16_t data_reg1 =
      second_data_operand != nullptr ? second_data_operand->index : 0;
  const std::uint16_t destination_reg =
      destination_operand != nullptr ? destination_operand->index : 0;
  std::span<std::byte> lds_storage(
      state->lds_bytes.data(), state->lds_bytes.size());
  if (workgroup != nullptr && !workgroup->shared_lds.empty()) {
    lds_storage = workgroup->shared_lds;
  }
  for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
       ++lane_index) {
    if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
      continue;
    }

    const std::uint32_t base_address = state->vgprs[address_reg][lane_index];
    std::uint64_t lds_address0 = 0;
    if (!ComputeDsAddress(base_address, offset0_operand->imm32, offset_scale,
                          &lds_address0, error_message)) {
      return false;
    }
    std::uint64_t lds_address1 = 0;
    if ((is_pair_write || is_pair_read) &&
        !ComputeDsAddress(base_address, offset1_operand->imm32, offset_scale,
                          &lds_address1, error_message)) {
      return false;
    }

    if (is_pair_write) {
      if (!WriteLdsValue(lds_storage, lds_address0, access_size,
                         state->vgprs[data_reg][lane_index], error_message) ||
          !WriteLdsValue(lds_storage, lds_address1, access_size,
                         state->vgprs[data_reg1][lane_index], error_message)) {
        return false;
      }
      continue;
    }

    if (is_pair_read) {
      std::uint32_t lds_value0 = 0;
      std::uint32_t lds_value1 = 0;
      if (!ReadLdsValue(lds_storage, lds_address0, access_size, false,
                        &lds_value0, error_message) ||
          !ReadLdsValue(lds_storage, lds_address1, access_size, false,
                        &lds_value1, error_message)) {
        return false;
      }
      state->vgprs[destination_reg][lane_index] = lds_value0;
      state->vgprs[destination_reg + 1u][lane_index] = lds_value1;
      continue;
    }

    if (is_read) {
      std::uint32_t lds_value = 0;
      if (!ReadLdsValue(lds_storage, lds_address0, access_size, sign_extend,
                        &lds_value, error_message)) {
        return false;
      }
      state->vgprs[destination_reg][lane_index] = lds_value;
      continue;
    }

    if (instruction.opcode == CompiledOpcode::kDsWriteB32 ||
        instruction.opcode == CompiledOpcode::kDsWriteB8 ||
        instruction.opcode == CompiledOpcode::kDsWriteB16) {
      const std::uint32_t data_value = state->vgprs[data_reg][lane_index];
      if (!WriteLdsValue(lds_storage, lds_address0, access_size, data_value,
                         error_message)) {
        return false;
      }
      continue;
    }

    if (is_dual_data || is_dual_data_return) {
      const std::uint32_t data0_value = state->vgprs[data_reg][lane_index];
      const std::uint32_t data1_value = state->vgprs[data_reg1][lane_index];
      std::uint32_t lds_value = 0;
      if (!ReadLdsValue(lds_storage, lds_address0, access_size, false, &lds_value,
                        error_message)) {
        return false;
      }
      if (is_dual_data_return) {
        state->vgprs[destination_reg][lane_index] = lds_value;
      }
      lds_value = EvaluateDsDualDataUpdate(instruction.opcode, lds_value,
                                           data0_value, data1_value,
                                           error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      if (!WriteLdsValue(lds_storage, lds_address0, access_size, lds_value,
                         error_message)) {
        return false;
      }
      continue;
    }

    const std::uint32_t data_value = state->vgprs[data_reg][lane_index];
    std::uint32_t lds_value = 0;
    if (!ReadLdsValue(lds_storage, lds_address0, access_size, false, &lds_value,
                      error_message)) {
      return false;
    }
    if (is_return) {
      state->vgprs[destination_reg][lane_index] = lds_value;
    }
    lds_value = EvaluateDsUpdate(GetDsUpdateOpcode(instruction.opcode), lds_value,
                                 data_value, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    if (!WriteLdsValue(lds_storage, lds_address0, access_size, lds_value,
                       error_message)) {
      return false;
    }
  }

  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool Gfx950Interpreter::ExecuteGlobalAtomic(const DecodedInstruction& instruction,
                                            WaveExecutionState* state,
                                            ExecutionMemory* memory,
                                            std::string* error_message) const {
  if (memory == nullptr) {
    if (error_message != nullptr) {
      *error_message = "global atomic instruction requires execution memory";
    }
    return false;
  }

  const bool has_return = instruction.operand_count == 5;
  if (instruction.operand_count != 4 && instruction.operand_count != 5) {
    if (error_message != nullptr) {
      *error_message = "unexpected operand count";
    }
    return false;
  }

  const InstructionOperand* return_operand = has_return ? &instruction.operands[0]
                                                        : nullptr;
  const InstructionOperand& address_operand =
      instruction.operands[has_return ? 1 : 0];
  const InstructionOperand& data_operand =
      instruction.operands[has_return ? 2 : 1];
  const InstructionOperand& scalar_base_operand =
      instruction.operands[has_return ? 3 : 2];
  const InstructionOperand& offset_operand =
      instruction.operands[has_return ? 4 : 3];
  if (return_operand != nullptr && return_operand->kind != OperandKind::kVgpr) {
    if (error_message != nullptr) {
      *error_message = "global atomic return operand must be a VGPR";
    }
    return false;
  }
  if (data_operand.kind != OperandKind::kVgpr) {
    if (error_message != nullptr) {
      *error_message = "global atomic data operand must be a VGPR";
    }
    return false;
  }
  if (offset_operand.kind != OperandKind::kImm32) {
    if (error_message != nullptr) {
      *error_message = "global atomic offset must be an immediate";
    }
    return false;
  }

  const std::uint8_t memory_dword_count =
      GetGlobalAtomicMemoryDwordCount(instruction.opcode);
  const std::uint8_t data_dword_count =
      GetGlobalAtomicDataDwordCount(instruction.opcode);
  if (data_operand.index + data_dword_count - 1 >= state->vgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "global atomic data register range out of bounds";
    }
    return false;
  }
  if (return_operand != nullptr &&
      return_operand->index + memory_dword_count - 1 >= state->vgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "global atomic return register range out of bounds";
    }
    return false;
  }
  const std::int32_t signed_offset =
      static_cast<std::int32_t>(offset_operand.imm32);

  for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
       ++lane_index) {
    if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
      continue;
    }

    std::uint64_t address = 0;
    if (!ResolveVectorMemoryAddress(address_operand, &scalar_base_operand,
                                    signed_offset, *state, lane_index, &address,
                                    error_message)) {
      return false;
    }

    std::array<std::uint32_t, 4> old_dwords{};
    std::array<std::uint32_t, 4> data_dwords{};
    std::array<std::uint32_t, 4> new_dwords{};
    for (std::uint8_t dword_index = 0; dword_index < memory_dword_count;
         ++dword_index) {
      const std::uint64_t byte_offset =
          static_cast<std::uint64_t>(dword_index) * 4u;
      if (address > std::numeric_limits<std::uint64_t>::max() - byte_offset) {
        if (error_message != nullptr) {
          *error_message = "global atomic address overflow";
        }
        return false;
      }
      if (!ReadMemoryU32(memory, address + byte_offset, &old_dwords[dword_index],
                         error_message)) {
        return false;
      }
      new_dwords[dword_index] = old_dwords[dword_index];
    }
    for (std::uint8_t dword_index = 0; dword_index < data_dword_count;
         ++dword_index) {
      data_dwords[dword_index] = ReadVectorOperand(
          InstructionOperand::Vgpr(
              static_cast<std::uint16_t>(data_operand.index + dword_index)),
          *state, lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
    }

    if (instruction.opcode == "GLOBAL_ATOMIC_SWAP") {
      new_dwords[0] = data_dwords[0];
    } else if (instruction.opcode == "GLOBAL_ATOMIC_CMPSWAP") {
      if (old_dwords[0] == data_dwords[0]) {
        new_dwords[0] = data_dwords[1];
      }
    } else if (instruction.opcode == "GLOBAL_ATOMIC_ADD") {
      new_dwords[0] = old_dwords[0] + data_dwords[0];
    } else if (instruction.opcode == "GLOBAL_ATOMIC_SUB") {
      new_dwords[0] = old_dwords[0] - data_dwords[0];
    } else if (instruction.opcode == "GLOBAL_ATOMIC_SMIN") {
      new_dwords[0] =
          BitCast<std::int32_t>(old_dwords[0]) < BitCast<std::int32_t>(data_dwords[0])
              ? old_dwords[0]
              : data_dwords[0];
    } else if (instruction.opcode == "GLOBAL_ATOMIC_UMIN") {
      new_dwords[0] = old_dwords[0] < data_dwords[0] ? old_dwords[0]
                                                     : data_dwords[0];
    } else if (instruction.opcode == "GLOBAL_ATOMIC_SMAX") {
      new_dwords[0] =
          BitCast<std::int32_t>(old_dwords[0]) > BitCast<std::int32_t>(data_dwords[0])
              ? old_dwords[0]
              : data_dwords[0];
    } else if (instruction.opcode == "GLOBAL_ATOMIC_UMAX") {
      new_dwords[0] = old_dwords[0] > data_dwords[0] ? old_dwords[0]
                                                     : data_dwords[0];
    } else if (instruction.opcode == "GLOBAL_ATOMIC_AND") {
      new_dwords[0] = old_dwords[0] & data_dwords[0];
    } else if (instruction.opcode == "GLOBAL_ATOMIC_OR") {
      new_dwords[0] = old_dwords[0] | data_dwords[0];
    } else if (instruction.opcode == "GLOBAL_ATOMIC_XOR") {
      new_dwords[0] = old_dwords[0] ^ data_dwords[0];
    } else if (instruction.opcode == "GLOBAL_ATOMIC_INC") {
      new_dwords[0] = AtomicIncU32(old_dwords[0], data_dwords[0]);
    } else if (instruction.opcode == "GLOBAL_ATOMIC_DEC") {
      new_dwords[0] = AtomicDecU32(old_dwords[0], data_dwords[0]);
    } else if (instruction.opcode == "GLOBAL_ATOMIC_ADD_F32") {
      new_dwords[0] =
          BitCast<std::uint32_t>(BitCast<float>(old_dwords[0]) +
                                 BitCast<float>(data_dwords[0]));
    } else if (instruction.opcode == "GLOBAL_ATOMIC_PK_ADD_F16") {
      new_dwords[0] = PackedHalfAdd(old_dwords[0], data_dwords[0]);
    } else if (instruction.opcode == "GLOBAL_ATOMIC_PK_ADD_BF16") {
      new_dwords[0] = PackedBFloat16Add(old_dwords[0], data_dwords[0]);
    } else if (instruction.opcode == "GLOBAL_ATOMIC_ADD_F64" ||
               instruction.opcode == "GLOBAL_ATOMIC_MIN_F64" ||
               instruction.opcode == "GLOBAL_ATOMIC_MAX_F64") {
      const std::uint64_t old_value = ComposeU64(old_dwords[0], old_dwords[1]);
      const std::uint64_t data_value = ComposeU64(data_dwords[0], data_dwords[1]);
      const double old_double = BitCast<double>(old_value);
      const double data_double = BitCast<double>(data_value);
      double new_double = old_double;
      if (instruction.opcode == "GLOBAL_ATOMIC_ADD_F64") {
        new_double = old_double + data_double;
      } else if (instruction.opcode == "GLOBAL_ATOMIC_MIN_F64") {
        new_double = std::fmin(old_double, data_double);
      } else {
        new_double = std::fmax(old_double, data_double);
      }
      const std::uint64_t new_value = BitCast<std::uint64_t>(new_double);
      SplitU64(new_value, &new_dwords[0], &new_dwords[1]);
    } else if (instruction.opcode == "GLOBAL_ATOMIC_SWAP_X2" ||
               instruction.opcode == "GLOBAL_ATOMIC_CMPSWAP_X2" ||
               instruction.opcode == "GLOBAL_ATOMIC_ADD_X2" ||
               instruction.opcode == "GLOBAL_ATOMIC_SUB_X2" ||
               instruction.opcode == "GLOBAL_ATOMIC_SMIN_X2" ||
               instruction.opcode == "GLOBAL_ATOMIC_UMIN_X2" ||
               instruction.opcode == "GLOBAL_ATOMIC_SMAX_X2" ||
               instruction.opcode == "GLOBAL_ATOMIC_UMAX_X2" ||
               instruction.opcode == "GLOBAL_ATOMIC_AND_X2" ||
               instruction.opcode == "GLOBAL_ATOMIC_OR_X2" ||
               instruction.opcode == "GLOBAL_ATOMIC_XOR_X2" ||
               instruction.opcode == "GLOBAL_ATOMIC_INC_X2" ||
               instruction.opcode == "GLOBAL_ATOMIC_DEC_X2") {
      const std::uint64_t old_value = ComposeU64(old_dwords[0], old_dwords[1]);
      const std::uint64_t data_value = ComposeU64(data_dwords[0], data_dwords[1]);
      std::uint64_t new_value = old_value;
      if (instruction.opcode == "GLOBAL_ATOMIC_SWAP_X2") {
        new_value = data_value;
      } else if (instruction.opcode == "GLOBAL_ATOMIC_CMPSWAP_X2") {
        const std::uint64_t replacement_value =
            ComposeU64(data_dwords[2], data_dwords[3]);
        if (old_value == data_value) {
          new_value = replacement_value;
        }
      } else if (instruction.opcode == "GLOBAL_ATOMIC_ADD_X2") {
        new_value = old_value + data_value;
      } else if (instruction.opcode == "GLOBAL_ATOMIC_SUB_X2") {
        new_value = old_value - data_value;
      } else if (instruction.opcode == "GLOBAL_ATOMIC_SMIN_X2") {
        new_value =
            BitCast<std::int64_t>(old_value) < BitCast<std::int64_t>(data_value)
                ? old_value
                : data_value;
      } else if (instruction.opcode == "GLOBAL_ATOMIC_UMIN_X2") {
        new_value = old_value < data_value ? old_value : data_value;
      } else if (instruction.opcode == "GLOBAL_ATOMIC_SMAX_X2") {
        new_value =
            BitCast<std::int64_t>(old_value) > BitCast<std::int64_t>(data_value)
                ? old_value
                : data_value;
      } else if (instruction.opcode == "GLOBAL_ATOMIC_UMAX_X2") {
        new_value = old_value > data_value ? old_value : data_value;
      } else if (instruction.opcode == "GLOBAL_ATOMIC_AND_X2") {
        new_value = old_value & data_value;
      } else if (instruction.opcode == "GLOBAL_ATOMIC_OR_X2") {
        new_value = old_value | data_value;
      } else if (instruction.opcode == "GLOBAL_ATOMIC_XOR_X2") {
        new_value = old_value ^ data_value;
      } else if (instruction.opcode == "GLOBAL_ATOMIC_INC_X2") {
        new_value = AtomicIncU64(old_value, data_value);
      } else if (instruction.opcode == "GLOBAL_ATOMIC_DEC_X2") {
        new_value = AtomicDecU64(old_value, data_value);
      }
      SplitU64(new_value, &new_dwords[0], &new_dwords[1]);
    } else {
      if (error_message != nullptr) {
        *error_message = "unsupported global atomic opcode";
      }
      return false;
    }

    for (std::uint8_t dword_index = 0; dword_index < memory_dword_count;
         ++dword_index) {
      const std::uint64_t byte_offset =
          static_cast<std::uint64_t>(dword_index) * 4u;
      if (!WriteMemoryU32(memory, address + byte_offset, new_dwords[dword_index],
                          error_message)) {
        return false;
      }
      if (return_operand != nullptr &&
          !WriteVectorOperand(
              InstructionOperand::Vgpr(static_cast<std::uint16_t>(
                  return_operand->index + dword_index)),
              lane_index, old_dwords[dword_index], state, error_message)) {
        return false;
      }
    }
  }
  return true;
}

bool Gfx950Interpreter::ExecuteGlobalAtomic(const CompiledInstruction& instruction,
                                            WaveExecutionState* state,
                                            ExecutionMemory* memory,
                                            std::string* error_message) const {
  if (memory == nullptr) {
    if (error_message != nullptr) {
      *error_message = "global atomic instruction requires execution memory";
    }
    return false;
  }

  const bool has_return = instruction.HasReturn();
  if (instruction.operand_count != (has_return ? 5 : 4)) {
    if (error_message != nullptr) {
      *error_message = "unexpected operand count";
    }
    return false;
  }

  const InstructionOperand* return_operand = has_return ? &instruction.operands[0]
                                                        : nullptr;
  const InstructionOperand& address_operand =
      instruction.operands[has_return ? 1 : 0];
  const InstructionOperand& data_operand =
      instruction.operands[has_return ? 2 : 1];
  const InstructionOperand& scalar_base_operand =
      instruction.operands[has_return ? 3 : 2];
  const InstructionOperand& offset_operand =
      instruction.operands[has_return ? 4 : 3];
  if (address_operand.kind != OperandKind::kVgpr) {
    if (error_message != nullptr) {
      *error_message = "global atomic address operand must be a VGPR pair";
    }
    return false;
  }
  if (return_operand != nullptr && return_operand->kind != OperandKind::kVgpr) {
    if (error_message != nullptr) {
      *error_message = "global atomic return operand must be a VGPR";
    }
    return false;
  }
  if (data_operand.kind != OperandKind::kVgpr) {
    if (error_message != nullptr) {
      *error_message = "global atomic data operand must be a VGPR";
    }
    return false;
  }
  if (scalar_base_operand.kind != OperandKind::kSgpr) {
    if (error_message != nullptr) {
      *error_message = "global atomic scalar base must be an SGPR pair";
    }
    return false;
  }
  if (offset_operand.kind != OperandKind::kImm32) {
    if (error_message != nullptr) {
      *error_message = "global atomic offset must be an immediate";
    }
    return false;
  }

  const std::uint8_t memory_dword_count = instruction.memory_dword_count;
  const std::uint8_t data_dword_count = instruction.data_dword_count;
  if (memory_dword_count == 0 || data_dword_count == 0) {
    if (error_message != nullptr) {
      *error_message = "global atomic metadata is incomplete";
    }
    return false;
  }
  if (address_operand.index + 1 >= state->vgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "global atomic address register pair out of bounds";
    }
    return false;
  }
  if (data_operand.index + data_dword_count - 1 >= state->vgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "global atomic data register range out of bounds";
    }
    return false;
  }
  if (scalar_base_operand.index + 1 >= state->sgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "global atomic scalar base out of bounds";
    }
    return false;
  }
  if (return_operand != nullptr &&
      return_operand->index + memory_dword_count - 1 >= state->vgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "global atomic return register range out of bounds";
    }
    return false;
  }
  const std::int32_t signed_offset =
      static_cast<std::int32_t>(offset_operand.imm32);
  const std::uint16_t address_reg = address_operand.index;
  const std::uint16_t data_reg = data_operand.index;
  const std::uint16_t return_reg =
      return_operand != nullptr ? return_operand->index : 0;
  const std::uint64_t scalar_base =
      ComposeU64(state->sgprs[scalar_base_operand.index],
                 state->sgprs[scalar_base_operand.index + 1]);
  const auto resolve_address = [&](std::size_t lane_index,
                                   std::uint64_t* address) -> bool {
    if (address == nullptr) {
      if (error_message != nullptr) {
        *error_message = "global atomic address output must not be null";
      }
      return false;
    }

    *address = ComposeU64(state->vgprs[address_reg][lane_index],
                          state->vgprs[address_reg + 1][lane_index]);
    if (*address > std::numeric_limits<std::uint64_t>::max() - scalar_base) {
      if (error_message != nullptr) {
        *error_message = "global address overflow";
      }
      return false;
    }
    *address += scalar_base;
    if (signed_offset < 0) {
      const std::uint64_t magnitude =
          static_cast<std::uint64_t>(-signed_offset);
      if (*address < magnitude) {
        if (error_message != nullptr) {
          *error_message = "vector memory address underflow";
        }
        return false;
      }
      *address -= magnitude;
    } else {
      const std::uint64_t magnitude =
          static_cast<std::uint64_t>(signed_offset);
      if (*address > std::numeric_limits<std::uint64_t>::max() - magnitude) {
        if (error_message != nullptr) {
          *error_message = "vector memory address overflow";
        }
        return false;
      }
      *address += magnitude;
    }
    return true;
  };

  if (!has_return && memory_dword_count == 1 && data_dword_count == 1) {
    for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
         ++lane_index) {
      if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
        continue;
      }

      std::uint64_t address = 0;
      if (!resolve_address(lane_index, &address)) {
        return false;
      }

      std::uint32_t old_value = 0;
      if (!ReadMemoryU32(memory, address, &old_value, error_message)) {
        return false;
      }

      const std::uint32_t data_value = state->vgprs[data_reg][lane_index];
      std::uint32_t new_value = old_value;
      switch (instruction.opcode) {
        case CompiledOpcode::kGlobalAtomicSwap:
          new_value = data_value;
          break;
        case CompiledOpcode::kGlobalAtomicAdd:
          new_value = old_value + data_value;
          break;
        case CompiledOpcode::kGlobalAtomicSub:
          new_value = old_value - data_value;
          break;
        case CompiledOpcode::kGlobalAtomicSMin:
          new_value =
              BitCast<std::int32_t>(old_value) < BitCast<std::int32_t>(data_value)
                  ? old_value
                  : data_value;
          break;
        case CompiledOpcode::kGlobalAtomicUMin:
          new_value = old_value < data_value ? old_value : data_value;
          break;
        case CompiledOpcode::kGlobalAtomicSMax:
          new_value =
              BitCast<std::int32_t>(old_value) > BitCast<std::int32_t>(data_value)
                  ? old_value
                  : data_value;
          break;
        case CompiledOpcode::kGlobalAtomicUMax:
          new_value = old_value > data_value ? old_value : data_value;
          break;
        case CompiledOpcode::kGlobalAtomicAnd:
          new_value = old_value & data_value;
          break;
        case CompiledOpcode::kGlobalAtomicOr:
          new_value = old_value | data_value;
          break;
        case CompiledOpcode::kGlobalAtomicXor:
          new_value = old_value ^ data_value;
          break;
        case CompiledOpcode::kGlobalAtomicInc:
          new_value = AtomicIncU32(old_value, data_value);
          break;
        case CompiledOpcode::kGlobalAtomicDec:
          new_value = AtomicDecU32(old_value, data_value);
          break;
        case CompiledOpcode::kGlobalAtomicAddF32:
          new_value =
              BitCast<std::uint32_t>(BitCast<float>(old_value) +
                                     BitCast<float>(data_value));
          break;
        case CompiledOpcode::kGlobalAtomicPkAddF16:
          new_value = PackedHalfAdd(old_value, data_value);
          break;
        case CompiledOpcode::kGlobalAtomicPkAddBf16:
          new_value = PackedBFloat16Add(old_value, data_value);
          break;
        default:
          break;
      }

      if (!WriteMemoryU32(memory, address, new_value, error_message)) {
        return false;
      }
    }
    return true;
  }

  for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
       ++lane_index) {
    if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
      continue;
    }

    std::uint64_t address =
        0;
    if (!resolve_address(lane_index, &address)) {
      return false;
    }

    std::array<std::uint32_t, 4> old_dwords{};
    std::array<std::uint32_t, 4> data_dwords{};
    std::array<std::uint32_t, 4> new_dwords{};
    for (std::uint8_t dword_index = 0; dword_index < memory_dword_count;
         ++dword_index) {
      const std::uint64_t byte_offset =
          static_cast<std::uint64_t>(dword_index) * 4u;
      if (address > std::numeric_limits<std::uint64_t>::max() - byte_offset) {
        if (error_message != nullptr) {
          *error_message = "global atomic address overflow";
        }
        return false;
      }
      if (!ReadMemoryU32(memory, address + byte_offset, &old_dwords[dword_index],
                         error_message)) {
        return false;
      }
      new_dwords[dword_index] = old_dwords[dword_index];
    }
    for (std::uint8_t dword_index = 0; dword_index < data_dword_count;
         ++dword_index) {
      data_dwords[dword_index] =
          state->vgprs[static_cast<std::uint16_t>(data_reg + dword_index)]
                      [lane_index];
    }

    switch (instruction.opcode) {
      case CompiledOpcode::kGlobalAtomicSwap:
        new_dwords[0] = data_dwords[0];
        break;
      case CompiledOpcode::kGlobalAtomicCmpSwap:
        if (old_dwords[0] == data_dwords[0]) {
          new_dwords[0] = data_dwords[1];
        }
        break;
      case CompiledOpcode::kGlobalAtomicAdd:
        new_dwords[0] = old_dwords[0] + data_dwords[0];
        break;
      case CompiledOpcode::kGlobalAtomicSub:
        new_dwords[0] = old_dwords[0] - data_dwords[0];
        break;
      case CompiledOpcode::kGlobalAtomicSMin:
        new_dwords[0] =
            BitCast<std::int32_t>(old_dwords[0]) < BitCast<std::int32_t>(data_dwords[0])
                ? old_dwords[0]
                : data_dwords[0];
        break;
      case CompiledOpcode::kGlobalAtomicUMin:
        new_dwords[0] =
            old_dwords[0] < data_dwords[0] ? old_dwords[0] : data_dwords[0];
        break;
      case CompiledOpcode::kGlobalAtomicSMax:
        new_dwords[0] =
            BitCast<std::int32_t>(old_dwords[0]) > BitCast<std::int32_t>(data_dwords[0])
                ? old_dwords[0]
                : data_dwords[0];
        break;
      case CompiledOpcode::kGlobalAtomicUMax:
        new_dwords[0] =
            old_dwords[0] > data_dwords[0] ? old_dwords[0] : data_dwords[0];
        break;
      case CompiledOpcode::kGlobalAtomicAnd:
        new_dwords[0] = old_dwords[0] & data_dwords[0];
        break;
      case CompiledOpcode::kGlobalAtomicOr:
        new_dwords[0] = old_dwords[0] | data_dwords[0];
        break;
      case CompiledOpcode::kGlobalAtomicXor:
        new_dwords[0] = old_dwords[0] ^ data_dwords[0];
        break;
      case CompiledOpcode::kGlobalAtomicInc:
        new_dwords[0] = AtomicIncU32(old_dwords[0], data_dwords[0]);
        break;
      case CompiledOpcode::kGlobalAtomicDec:
        new_dwords[0] = AtomicDecU32(old_dwords[0], data_dwords[0]);
        break;
      case CompiledOpcode::kGlobalAtomicAddF32:
        new_dwords[0] =
            BitCast<std::uint32_t>(BitCast<float>(old_dwords[0]) +
                                   BitCast<float>(data_dwords[0]));
        break;
      case CompiledOpcode::kGlobalAtomicPkAddF16:
        new_dwords[0] = PackedHalfAdd(old_dwords[0], data_dwords[0]);
        break;
      case CompiledOpcode::kGlobalAtomicPkAddBf16:
        new_dwords[0] = PackedBFloat16Add(old_dwords[0], data_dwords[0]);
        break;
      case CompiledOpcode::kGlobalAtomicAddF64:
      case CompiledOpcode::kGlobalAtomicMinF64:
      case CompiledOpcode::kGlobalAtomicMaxF64: {
        const std::uint64_t old_value = ComposeU64(old_dwords[0], old_dwords[1]);
        const std::uint64_t data_value = ComposeU64(data_dwords[0], data_dwords[1]);
        const double old_double = BitCast<double>(old_value);
        const double data_double = BitCast<double>(data_value);
        double new_double = old_double;
        if (instruction.opcode == CompiledOpcode::kGlobalAtomicAddF64) {
          new_double = old_double + data_double;
        } else if (instruction.opcode == CompiledOpcode::kGlobalAtomicMinF64) {
          new_double = std::fmin(old_double, data_double);
        } else {
          new_double = std::fmax(old_double, data_double);
        }
        const std::uint64_t new_value = BitCast<std::uint64_t>(new_double);
        SplitU64(new_value, &new_dwords[0], &new_dwords[1]);
        break;
      }
      case CompiledOpcode::kGlobalAtomicSwapX2:
      case CompiledOpcode::kGlobalAtomicCmpSwapX2:
      case CompiledOpcode::kGlobalAtomicAddX2:
      case CompiledOpcode::kGlobalAtomicSubX2:
      case CompiledOpcode::kGlobalAtomicSMinX2:
      case CompiledOpcode::kGlobalAtomicUMinX2:
      case CompiledOpcode::kGlobalAtomicSMaxX2:
      case CompiledOpcode::kGlobalAtomicUMaxX2:
      case CompiledOpcode::kGlobalAtomicAndX2:
      case CompiledOpcode::kGlobalAtomicOrX2:
      case CompiledOpcode::kGlobalAtomicXorX2:
      case CompiledOpcode::kGlobalAtomicIncX2:
      case CompiledOpcode::kGlobalAtomicDecX2: {
        const std::uint64_t old_value = ComposeU64(old_dwords[0], old_dwords[1]);
        const std::uint64_t data_value = ComposeU64(data_dwords[0], data_dwords[1]);
        std::uint64_t new_value = old_value;
        if (instruction.opcode == CompiledOpcode::kGlobalAtomicSwapX2) {
          new_value = data_value;
        } else if (instruction.opcode == CompiledOpcode::kGlobalAtomicCmpSwapX2) {
          const std::uint64_t replacement_value =
              ComposeU64(data_dwords[2], data_dwords[3]);
          if (old_value == data_value) {
            new_value = replacement_value;
          }
        } else if (instruction.opcode == CompiledOpcode::kGlobalAtomicAddX2) {
          new_value = old_value + data_value;
        } else if (instruction.opcode == CompiledOpcode::kGlobalAtomicSubX2) {
          new_value = old_value - data_value;
        } else if (instruction.opcode == CompiledOpcode::kGlobalAtomicSMinX2) {
          new_value =
              BitCast<std::int64_t>(old_value) < BitCast<std::int64_t>(data_value)
                  ? old_value
                  : data_value;
        } else if (instruction.opcode == CompiledOpcode::kGlobalAtomicUMinX2) {
          new_value = old_value < data_value ? old_value : data_value;
        } else if (instruction.opcode == CompiledOpcode::kGlobalAtomicSMaxX2) {
          new_value =
              BitCast<std::int64_t>(old_value) > BitCast<std::int64_t>(data_value)
                  ? old_value
                  : data_value;
        } else if (instruction.opcode == CompiledOpcode::kGlobalAtomicUMaxX2) {
          new_value = old_value > data_value ? old_value : data_value;
        } else if (instruction.opcode == CompiledOpcode::kGlobalAtomicAndX2) {
          new_value = old_value & data_value;
        } else if (instruction.opcode == CompiledOpcode::kGlobalAtomicOrX2) {
          new_value = old_value | data_value;
        } else if (instruction.opcode == CompiledOpcode::kGlobalAtomicXorX2) {
          new_value = old_value ^ data_value;
        } else if (instruction.opcode == CompiledOpcode::kGlobalAtomicIncX2) {
          new_value = AtomicIncU64(old_value, data_value);
        } else {
          new_value = AtomicDecU64(old_value, data_value);
        }
        SplitU64(new_value, &new_dwords[0], &new_dwords[1]);
        break;
      }
      default:
        if (error_message != nullptr) {
          *error_message = "unsupported compiled global atomic opcode";
        }
        return false;
    }

    for (std::uint8_t dword_index = 0; dword_index < memory_dword_count;
         ++dword_index) {
      const std::uint64_t byte_offset =
          static_cast<std::uint64_t>(dword_index) * 4u;
      if (!WriteMemoryU32(memory, address + byte_offset, new_dwords[dword_index],
                          error_message)) {
        return false;
      }
      if (return_operand != nullptr) {
        state->vgprs[static_cast<std::uint16_t>(return_reg + dword_index)]
                    [lane_index] = old_dwords[dword_index];
      }
    }
  }
  return true;
}

bool Gfx950Interpreter::ExecuteBarrier(const DecodedInstruction& instruction,
                                       WaveExecutionState* state,
                                       const WorkgroupExecutionContext* workgroup,
                                       bool* wave_yielded,
                                       std::string* error_message) const {
  if (!ValidateOperandCount(instruction, 0, error_message)) {
    return false;
  }

  state->waiting_on_barrier = false;
  const std::uint32_t wave_count =
      workgroup != nullptr ? workgroup->wave_count : state->workgroup_wave_count;
  if (wave_count <= 1) {
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }

  state->waiting_on_barrier = true;
  if (wave_yielded != nullptr) {
    *wave_yielded = true;
  }
  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool Gfx950Interpreter::ExecuteBarrier(const CompiledInstruction& instruction,
                                       WaveExecutionState* state,
                                       const WorkgroupExecutionContext* workgroup,
                                       bool* wave_yielded,
                                       std::string* error_message) const {
  if (!ValidateOperandCount(instruction, 0, error_message)) {
    return false;
  }

  state->waiting_on_barrier = false;
  const std::uint32_t wave_count =
      workgroup != nullptr ? workgroup->wave_count : state->workgroup_wave_count;
  if (wave_count <= 1) {
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }

  state->waiting_on_barrier = true;
  if (wave_yielded != nullptr) {
    *wave_yielded = true;
  }
  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool Gfx950Interpreter::ExecuteBranch(const DecodedInstruction& instruction,
                                      WaveExecutionState* state,
                                      bool* pc_was_updated,
                                      std::string* error_message) const {
  if (!ValidateOperandCount(instruction, 1, error_message)) {
    return false;
  }
  if (instruction.operands[0].kind != OperandKind::kImm32) {
    if (error_message != nullptr) {
      *error_message = "branch operand must be an immediate offset";
    }
    return false;
  }

  const std::int32_t delta =
      static_cast<std::int32_t>(instruction.operands[0].imm32);

  if (instruction.opcode == "S_BRANCH") {
    return ApplyRelativeBranch(delta, state, pc_was_updated, error_message);
  }
  if (instruction.opcode == "S_CBRANCH_SCC0") {
    if (!state->scc) {
      return ApplyRelativeBranch(delta, state, pc_was_updated, error_message);
    }
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }
  if (instruction.opcode == "S_CBRANCH_SCC1") {
    if (state->scc) {
      return ApplyRelativeBranch(delta, state, pc_was_updated, error_message);
    }
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }
  if (instruction.opcode == "S_CBRANCH_VCCZ") {
    if (state->vcc_mask == 0) {
      return ApplyRelativeBranch(delta, state, pc_was_updated, error_message);
    }
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }
  if (instruction.opcode == "S_CBRANCH_VCCNZ") {
    if (state->vcc_mask != 0) {
      return ApplyRelativeBranch(delta, state, pc_was_updated, error_message);
    }
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }
  if (instruction.opcode == "S_CBRANCH_EXECZ") {
    if (state->exec_mask == 0) {
      return ApplyRelativeBranch(delta, state, pc_was_updated, error_message);
    }
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }
  if (instruction.opcode == "S_CBRANCH_EXECNZ") {
    if (state->exec_mask != 0) {
      return ApplyRelativeBranch(delta, state, pc_was_updated, error_message);
    }
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }

  if (error_message != nullptr) {
    *error_message = "unsupported branch opcode";
  }
  return false;
}

bool Gfx950Interpreter::ExecuteBranch(const CompiledInstruction& instruction,
                                      WaveExecutionState* state,
                                      bool* pc_was_updated,
                                      std::string* error_message) const {
  if (!ValidateOperandCount(instruction, 1, error_message)) {
    return false;
  }
  if (instruction.operands[0].kind != OperandKind::kImm32) {
    if (error_message != nullptr) {
      *error_message = "branch operand must be an immediate offset";
    }
    return false;
  }

  const std::int32_t delta =
      static_cast<std::int32_t>(instruction.operands[0].imm32);

  switch (instruction.opcode) {
    case CompiledOpcode::kSBranch:
      return ApplyRelativeBranch(delta, state, pc_was_updated, error_message);
    case CompiledOpcode::kSCbranchScc0:
      if (!state->scc) {
        return ApplyRelativeBranch(delta, state, pc_was_updated, error_message);
      }
      if (error_message != nullptr) {
        error_message->clear();
      }
      return true;
    case CompiledOpcode::kSCbranchScc1:
      if (state->scc) {
        return ApplyRelativeBranch(delta, state, pc_was_updated, error_message);
      }
      if (error_message != nullptr) {
        error_message->clear();
      }
      return true;
    case CompiledOpcode::kSCbranchVccz:
      if (state->vcc_mask == 0) {
        return ApplyRelativeBranch(delta, state, pc_was_updated, error_message);
      }
      if (error_message != nullptr) {
        error_message->clear();
      }
      return true;
    case CompiledOpcode::kSCbranchVccnz:
      if (state->vcc_mask != 0) {
        return ApplyRelativeBranch(delta, state, pc_was_updated, error_message);
      }
      if (error_message != nullptr) {
        error_message->clear();
      }
      return true;
    case CompiledOpcode::kSCbranchExecz:
      if (state->exec_mask == 0) {
        return ApplyRelativeBranch(delta, state, pc_was_updated, error_message);
      }
      if (error_message != nullptr) {
        error_message->clear();
      }
      return true;
    case CompiledOpcode::kSCbranchExecnz:
      if (state->exec_mask != 0) {
        return ApplyRelativeBranch(delta, state, pc_was_updated, error_message);
      }
      if (error_message != nullptr) {
        error_message->clear();
      }
      return true;
    default:
      if (error_message != nullptr) {
        *error_message = "unsupported compiled branch opcode";
      }
      return false;
  }
}

bool Gfx950Interpreter::ValidateOperandCount(
    const DecodedInstruction& instruction,
    std::uint8_t expected_operands,
    std::string* error_message) const {
  if (instruction.operand_count == expected_operands) {
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }

  if (error_message != nullptr) {
    *error_message = "unexpected operand count";
  }
  return false;
}

bool Gfx950Interpreter::ValidateOperandCount(
    const CompiledInstruction& instruction,
    std::uint8_t expected_operands,
    std::string* error_message) const {
  if (instruction.operand_count == expected_operands) {
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }

  if (error_message != nullptr) {
    *error_message = "unexpected operand count";
  }
  return false;
}

std::uint32_t Gfx950Interpreter::ReadScalarOperand(
    const InstructionOperand& operand,
    const WaveExecutionState& state,
    std::string* error_message) const {
  if (error_message != nullptr) {
    error_message->clear();
  }

  if (operand.kind == OperandKind::kImm32) {
    return operand.imm32;
  }
  if (operand.kind != OperandKind::kSgpr) {
    if (error_message != nullptr) {
      *error_message = "expected scalar source operand";
    }
    return 0;
  }
  if (operand.index >= state.sgprs.size()) {
    if (operand.index == kSrcVcczSgprIndex) {
      return state.vcc_mask == 0 ? 1u : 0u;
    }
    if (operand.index == kSrcExeczSgprIndex) {
      return state.exec_mask == 0 ? 1u : 0u;
    }
    if (operand.index == kSrcSccSgprIndex) {
      return state.scc ? 1u : 0u;
    }
    if (error_message != nullptr) {
      *error_message = "scalar register index out of range";
    }
    return 0;
  }
  if (operand.index == kExecPairSgprIndex) {
    return static_cast<std::uint32_t>(state.exec_mask);
  }
  if (operand.index == kExecPairSgprIndex + 1) {
    return static_cast<std::uint32_t>(state.exec_mask >> 32);
  }
  return state.sgprs[operand.index];
}

std::uint64_t Gfx950Interpreter::ReadScalarPairOperand(
    const InstructionOperand& operand,
    const WaveExecutionState& state,
    std::string* error_message) const {
  if (error_message != nullptr) {
    error_message->clear();
  }

  if (operand.kind == OperandKind::kImm32) {
    return operand.imm32;
  }
  if (operand.kind != OperandKind::kSgpr) {
    if (error_message != nullptr) {
      *error_message = "expected scalar register pair source operand";
    }
    return 0;
  }
  if (operand.index + 1 >= state.sgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "scalar register pair out of range";
    }
    return 0;
  }
  if (operand.index == kExecPairSgprIndex) {
    return state.exec_mask;
  }
  return static_cast<std::uint64_t>(state.sgprs[operand.index]) |
         (static_cast<std::uint64_t>(state.sgprs[operand.index + 1]) << 32);
}

std::uint32_t Gfx950Interpreter::ReadVectorOperand(
    const InstructionOperand& operand,
    const WaveExecutionState& state,
    std::size_t lane_index,
    std::string* error_message) const {
  if (error_message != nullptr) {
    error_message->clear();
  }

  if (operand.kind == OperandKind::kImm32) {
    return operand.imm32;
  }
  if (operand.kind == OperandKind::kVgpr) {
    if (operand.index >= state.vgprs.size()) {
      if (error_message != nullptr) {
        *error_message = "vector register index out of range";
      }
      return 0;
    }
    return state.vgprs[operand.index][lane_index];
  }
  if (operand.kind == OperandKind::kSgpr) {
    return ReadScalarOperand(operand, state, error_message);
  }

  if (error_message != nullptr) {
    *error_message = "unsupported vector operand kind";
  }
  return 0;
}

bool Gfx950Interpreter::ReadScalarAddressOperand(
    const InstructionOperand& operand,
    const WaveExecutionState& state,
    std::uint64_t* value,
    std::string* error_message) const {
  if (value == nullptr) {
    if (error_message != nullptr) {
      *error_message = "scalar address output must not be null";
    }
    return false;
  }
  if (operand.kind != OperandKind::kSgpr) {
    if (error_message != nullptr) {
      *error_message = "expected scalar address register operand";
    }
    return false;
  }
  if (operand.index + 1 >= state.sgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "scalar address register pair out of range";
    }
    return false;
  }

  *value = static_cast<std::uint64_t>(state.sgprs[operand.index]) |
           (static_cast<std::uint64_t>(state.sgprs[operand.index + 1]) << 32);
  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool Gfx950Interpreter::ReadVectorAddressOperand(
    const InstructionOperand& operand,
    const WaveExecutionState& state,
    std::size_t lane_index,
    std::uint64_t* value,
    std::string* error_message) const {
  if (value == nullptr) {
    if (error_message != nullptr) {
      *error_message = "vector address output must not be null";
    }
    return false;
  }
  if (operand.kind != OperandKind::kVgpr) {
    if (error_message != nullptr) {
      *error_message = "expected vector address register operand";
    }
    return false;
  }
  if (operand.index + 1 >= state.vgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "vector address register pair out of range";
    }
    return false;
  }

  *value = static_cast<std::uint64_t>(state.vgprs[operand.index][lane_index]) |
           (static_cast<std::uint64_t>(
                state.vgprs[operand.index + 1][lane_index])
            << 32);
  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool Gfx950Interpreter::ResolveVectorMemoryAddress(
    const InstructionOperand& address_operand,
    const InstructionOperand* scalar_base_operand,
    std::int32_t signed_offset,
    const WaveExecutionState& state,
    std::size_t lane_index,
    std::uint64_t* address,
    std::string* error_message) const {
  if (address == nullptr) {
    if (error_message != nullptr) {
      *error_message = "vector memory address output must not be null";
    }
    return false;
  }
  if (!ReadVectorAddressOperand(address_operand, state, lane_index, address,
                                error_message)) {
    return false;
  }

  if (scalar_base_operand != nullptr) {
    std::uint64_t scalar_base = 0;
    if (!ReadScalarAddressOperand(*scalar_base_operand, state, &scalar_base,
                                  error_message)) {
      return false;
    }
    if (*address > std::numeric_limits<std::uint64_t>::max() - scalar_base) {
      if (error_message != nullptr) {
        *error_message = "global address overflow";
      }
      return false;
    }
    *address += scalar_base;
  }

  if (signed_offset < 0) {
    const std::uint64_t magnitude =
        static_cast<std::uint64_t>(-signed_offset);
    if (*address < magnitude) {
      if (error_message != nullptr) {
        *error_message = "vector memory address underflow";
      }
      return false;
    }
    *address -= magnitude;
  } else {
    const std::uint64_t magnitude =
        static_cast<std::uint64_t>(signed_offset);
    if (*address > std::numeric_limits<std::uint64_t>::max() - magnitude) {
      if (error_message != nullptr) {
        *error_message = "vector memory address overflow";
      }
      return false;
    }
    *address += magnitude;
  }

  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool Gfx950Interpreter::WriteScalarOperand(const InstructionOperand& operand,
                                           std::uint32_t value,
                                           WaveExecutionState* state,
                                           std::string* error_message) const {
  if (operand.kind != OperandKind::kSgpr) {
    if (error_message != nullptr) {
      *error_message = "expected scalar destination operand";
    }
    return false;
  }
  if (operand.index >= state->sgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "scalar destination out of range";
    }
    return false;
  }

  if (error_message != nullptr) {
    error_message->clear();
  }
  state->sgprs[operand.index] = value;
  if (operand.index == kExecPairSgprIndex) {
    state->exec_mask =
        (state->exec_mask & 0xffffffff00000000ULL) | value;
  } else if (operand.index == kExecPairSgprIndex + 1) {
    state->exec_mask =
        (state->exec_mask & 0x00000000ffffffffULL) |
        (static_cast<std::uint64_t>(value) << 32);
  }
  return true;
}

bool Gfx950Interpreter::WriteScalarPairOperand(const InstructionOperand& operand,
                                               std::uint64_t value,
                                               WaveExecutionState* state,
                                               std::string* error_message) const {
  if (operand.kind != OperandKind::kSgpr) {
    if (error_message != nullptr) {
      *error_message = "expected scalar destination pair operand";
    }
    return false;
  }
  if (operand.index + 1 >= state->sgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "scalar destination pair out of range";
    }
    return false;
  }

  if (error_message != nullptr) {
    error_message->clear();
  }
  state->sgprs[operand.index] = static_cast<std::uint32_t>(value);
  state->sgprs[operand.index + 1] =
      static_cast<std::uint32_t>(value >> 32);
  if (operand.index == kExecPairSgprIndex) {
    state->exec_mask = value;
  }
  return true;
}

bool Gfx950Interpreter::WriteVectorOperand(const InstructionOperand& operand,
                                           std::size_t lane_index,
                                           std::uint32_t value,
                                           WaveExecutionState* state,
                                           std::string* error_message) const {
  if (operand.kind != OperandKind::kVgpr) {
    if (error_message != nullptr) {
      *error_message = "expected vector destination operand";
    }
    return false;
  }
  if (operand.index >= state->vgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "vector destination out of range";
    }
    return false;
  }

  if (error_message != nullptr) {
    error_message->clear();
  }
  state->vgprs[operand.index][lane_index] = value;
  return true;
}

bool Gfx950Interpreter::ApplyRelativeBranch(std::int32_t delta_in_instructions,
                                            WaveExecutionState* state,
                                            bool* pc_was_updated,
                                            std::string* error_message) const {
  const std::int64_t target_pc =
      static_cast<std::int64_t>(state->pc) + 1 +
      static_cast<std::int64_t>(delta_in_instructions);
  if (target_pc < 0) {
    if (error_message != nullptr) {
      *error_message = "branch target underflow";
    }
    return false;
  }

  if (error_message != nullptr) {
    error_message->clear();
  }
  state->pc = static_cast<std::uint64_t>(target_pc);
  if (pc_was_updated != nullptr) {
    *pc_was_updated = true;
  }
  return true;
}

bool Gfx950Interpreter::ReadMemoryU32(ExecutionMemory* memory,
                                      std::uint64_t address,
                                      std::uint32_t* value,
                                      std::string* error_message) const {
  if (memory == nullptr || value == nullptr) {
    if (error_message != nullptr) {
      *error_message = "memory load requires valid inputs";
    }
    return false;
  }

  if (!memory->LoadU32(address, value)) {
    if (error_message != nullptr) {
      *error_message = "memory load failed";
    }
    return false;
  }
  return true;
}

bool Gfx950Interpreter::ReadMemoryU16(ExecutionMemory* memory,
                                      std::uint64_t address,
                                      std::uint16_t* value,
                                      std::string* error_message) const {
  if (memory == nullptr || value == nullptr) {
    if (error_message != nullptr) {
      *error_message = "memory load requires valid inputs";
    }
    return false;
  }

  if (!memory->LoadU16(address, value)) {
    if (error_message != nullptr) {
      *error_message = "memory load failed";
    }
    return false;
  }
  return true;
}

bool Gfx950Interpreter::ReadMemoryU8(ExecutionMemory* memory,
                                     std::uint64_t address,
                                     std::uint8_t* value,
                                     std::string* error_message) const {
  if (memory == nullptr || value == nullptr) {
    if (error_message != nullptr) {
      *error_message = "memory load requires valid inputs";
    }
    return false;
  }

  if (!memory->LoadU8(address, value)) {
    if (error_message != nullptr) {
      *error_message = "memory load failed";
    }
    return false;
  }
  return true;
}

bool Gfx950Interpreter::WriteMemoryU16(ExecutionMemory* memory,
                                       std::uint64_t address,
                                       std::uint16_t value,
                                       std::string* error_message) const {
  if (memory == nullptr) {
    if (error_message != nullptr) {
      *error_message = "memory store requires valid memory";
    }
    return false;
  }

  if (!memory->StoreU16(address, value)) {
    if (error_message != nullptr) {
      *error_message = "memory store failed";
    }
    return false;
  }
  return true;
}

bool Gfx950Interpreter::WriteMemoryU8(ExecutionMemory* memory,
                                      std::uint64_t address,
                                      std::uint8_t value,
                                      std::string* error_message) const {
  if (memory == nullptr) {
    if (error_message != nullptr) {
      *error_message = "memory store requires valid memory";
    }
    return false;
  }

  if (!memory->StoreU8(address, value)) {
    if (error_message != nullptr) {
      *error_message = "memory store failed";
    }
    return false;
  }
  return true;
}

bool Gfx950Interpreter::WriteMemoryU32(ExecutionMemory* memory,
                                       std::uint64_t address,
                                       std::uint32_t value,
                                       std::string* error_message) const {
  if (memory == nullptr) {
    if (error_message != nullptr) {
      *error_message = "memory store requires valid memory";
    }
    return false;
  }

  if (!memory->StoreU32(address, value)) {
    if (error_message != nullptr) {
      *error_message = "memory store failed";
    }
    return false;
  }
  return true;
}

}  // namespace mirage::sim::isa
