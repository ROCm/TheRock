#include "lib/sim/isa/gfx1201/interpreter.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include "lib/sim/isa/gfx1201/support_catalog.h"

namespace mirage::sim::isa {
namespace {

template <typename To, typename From>
To BitCast(From value) {
  static_assert(sizeof(To) == sizeof(From));
  To result;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

constexpr std::uint32_t kGfx1201LaneCount = 32;
constexpr std::uint16_t kExecPairSgprIndex = 126;
constexpr std::uint16_t kImplicitVccPairSgprIndex = 248;
constexpr std::uint16_t kSrcVcczSgprIndex = 251;
constexpr std::uint16_t kSrcExeczSgprIndex = 252;
constexpr std::uint16_t kSrcSccSgprIndex = 253;

constexpr std::array<std::string_view, 195> kExecutableSeedOpcodes{{
    "S_ENDPGM",
    "S_NOP",
    "S_ADD_U32",
    "S_ADD_I32",
    "S_SUB_U32",
    "S_CMP_EQ_I32",
    "S_CMP_LG_I32",
    "S_CMP_GT_I32",
    "S_CMP_EQ_U32",
    "S_CMP_LG_U32",
    "S_CMP_GE_I32",
    "S_CMP_LT_I32",
    "S_CMP_LE_I32",
    "S_CMP_GT_U32",
    "S_CMP_GE_U32",
    "S_CMP_LT_U32",
    "S_CMP_LE_U32",
    "S_BRANCH",
    "S_CBRANCH_SCC0",
    "S_CBRANCH_SCC1",
    "S_CBRANCH_VCCZ",
    "S_CBRANCH_VCCNZ",
    "S_CBRANCH_EXECZ",
    "S_CBRANCH_EXECNZ",
    "S_MOV_B32",
    "S_MOVK_I32",
    "V_MOV_B32",
    "V_CMP_EQ_I32",
    "V_CMP_NE_I32",
    "V_CMP_LT_I32",
    "V_CMP_LE_I32",
    "V_CMP_GT_I32",
    "V_CMP_GE_I32",
    "V_CMP_EQ_U32",
    "V_CMP_NE_U32",
    "V_CMP_LT_U32",
    "V_CMP_LE_U32",
    "V_CMP_GT_U32",
    "V_CMP_GE_U32",
    "V_CMPX_EQ_I32",
    "V_CMPX_NE_I32",
    "V_CMPX_LT_I32",
    "V_CMPX_LE_I32",
    "V_CMPX_GT_I32",
    "V_CMPX_GE_I32",
    "V_CMPX_EQ_U32",
    "V_CMPX_NE_U32",
    "V_CMPX_LT_U32",
    "V_CMPX_LE_U32",
    "V_CMPX_GT_U32",
    "V_CMPX_GE_U32",
    "V_CMP_EQ_I64",
    "V_CMP_NE_I64",
    "V_CMP_LT_I64",
    "V_CMP_LE_I64",
    "V_CMP_GT_I64",
    "V_CMP_GE_I64",
    "V_CMP_EQ_U64",
    "V_CMP_NE_U64",
    "V_CMP_LT_U64",
    "V_CMP_LE_U64",
    "V_CMP_GT_U64",
    "V_CMP_GE_U64",
    "V_CMPX_EQ_I64",
    "V_CMPX_NE_I64",
    "V_CMPX_LT_I64",
    "V_CMPX_LE_I64",
    "V_CMPX_GT_I64",
    "V_CMPX_GE_I64",
    "V_CMPX_EQ_U64",
    "V_CMPX_NE_U64",
    "V_CMPX_LT_U64",
    "V_CMPX_LE_U64",
    "V_CMPX_GT_U64",
    "V_CMPX_GE_U64",
    "V_CMP_CLASS_F32",
    "V_CMP_EQ_F32",
    "V_CMP_GE_F32",
    "V_CMP_GT_F32",
    "V_CMP_LE_F32",
    "V_CMP_LG_F32",
    "V_CMP_LT_F32",
    "V_CMP_NEQ_F32",
    "V_CMP_O_F32",
    "V_CMP_U_F32",
    "V_CMP_NGE_F32",
    "V_CMP_NLG_F32",
    "V_CMP_NGT_F32",
    "V_CMP_NLE_F32",
    "V_CMP_NLT_F32",
    "V_CMPX_CLASS_F32",
    "V_CMPX_EQ_F32",
    "V_CMPX_GE_F32",
    "V_CMPX_GT_F32",
    "V_CMPX_LE_F32",
    "V_CMPX_LG_F32",
    "V_CMPX_LT_F32",
    "V_CMPX_NEQ_F32",
    "V_CMPX_O_F32",
    "V_CMPX_U_F32",
    "V_CMPX_NGE_F32",
    "V_CMPX_NLG_F32",
    "V_CMPX_NGT_F32",
    "V_CMPX_NLE_F32",
    "V_CMPX_NLT_F32",
    "V_CMP_CLASS_F64",
    "V_CMP_EQ_F64",
    "V_CMP_GE_F64",
    "V_CMP_GT_F64",
    "V_CMP_LE_F64",
    "V_CMP_LG_F64",
    "V_CMP_LT_F64",
    "V_CMP_NEQ_F64",
    "V_CMP_O_F64",
    "V_CMP_U_F64",
    "V_CMP_NGE_F64",
    "V_CMP_NLG_F64",
    "V_CMP_NGT_F64",
    "V_CMP_NLE_F64",
    "V_CMP_NLT_F64",
    "V_CMPX_CLASS_F64",
    "V_CMPX_EQ_F64",
    "V_CMPX_GE_F64",
    "V_CMPX_GT_F64",
    "V_CMPX_LE_F64",
    "V_CMPX_LG_F64",
    "V_CMPX_LT_F64",
    "V_CMPX_NEQ_F64",
    "V_CMPX_O_F64",
    "V_CMPX_U_F64",
    "V_CMPX_NGE_F64",
    "V_CMPX_NLG_F64",
    "V_CMPX_NGT_F64",
    "V_CMPX_NLE_F64",
    "V_CMPX_NLT_F64",
    "V_NOT_B32",
    "V_BFREV_B32",
    "V_CLS_I32",
    "V_CLZ_I32_U32",
    "V_CTZ_I32_B32",
    "V_CVT_F32_UBYTE0",
    "V_CVT_F32_UBYTE1",
    "V_CVT_F32_UBYTE2",
    "V_CVT_F32_UBYTE3",
    "V_CVT_F32_I32",
    "V_CVT_F32_U32",
    "V_CVT_F32_F64",
    "V_CVT_F64_F32",
    "V_CVT_F64_I32",
    "V_CVT_F64_U32",
    "V_CVT_U32_F32",
    "V_CVT_U32_F64",
    "V_CVT_I32_F32",
    "V_CVT_FLOOR_I32_F32",
    "V_CVT_NEAREST_I32_F32",
    "V_CVT_I32_F64",
    "V_EXP_F32",
    "V_LOG_F32",
    "V_RCP_F32",
    "V_RCP_IFLAG_F32",
    "V_RSQ_F32",
    "V_SQRT_F32",
    "V_SIN_F32",
    "V_COS_F32",
    "V_RCP_F64",
    "V_RSQ_F64",
    "V_SQRT_F64",
    "V_FREXP_EXP_I32_F32",
    "V_FREXP_MANT_F32",
    "V_FRACT_F32",
    "V_FREXP_EXP_I32_F64",
    "V_FREXP_MANT_F64",
    "V_FRACT_F64",
    "V_TRUNC_F32",
    "V_CEIL_F32",
    "V_RNDNE_F32",
    "V_FLOOR_F32",
    "V_TRUNC_F64",
    "V_CEIL_F64",
    "V_RNDNE_F64",
    "V_FLOOR_F64",
    "V_ADD_U32",
    "V_SUB_U32",
    "V_SUBREV_U32",
    "V_MIN_I32",
    "V_MAX_I32",
    "V_MIN_U32",
    "V_MAX_U32",
    "V_CNDMASK_B32",
    "V_LSHRREV_B32",
    "V_ASHRREV_I32",
    "V_LSHLREV_B32",
    "V_AND_B32",
    "V_OR_B32",
    "V_XOR_B32",
}};

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

std::int32_t FloorFloatToI32(float value) {
  return TruncateDoubleToI32(std::floor(static_cast<double>(value)));
}

std::int32_t RoundNearestFloatToI32(float value) {
  if (std::isnan(value)) {
    return 0;
  }
  const double input = static_cast<double>(value);
  const double floor_value = std::floor(input);
  const double fraction = input - floor_value;
  double rounded = floor_value;
  if (fraction > 0.5) {
    rounded = floor_value + 1.0;
  } else if (fraction == 0.5) {
    const auto floor_integer = static_cast<std::int64_t>(floor_value);
    rounded = (floor_integer & 1LL) == 0 ? floor_value : floor_value + 1.0;
  }
  return TruncateDoubleToI32(rounded);
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

float EvaluateUnaryFloatMathF32(std::string_view opcode, float input) {
  if (opcode == "V_EXP_F32") {
    return std::exp2(input);
  }
  if (opcode == "V_LOG_F32") {
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

double EvaluateUnaryFloatMathF64(std::string_view opcode, double input) {
  if (opcode == "V_RCP_F64") {
    return 1.0 / input;
  }
  if (opcode == "V_RSQ_F64") {
    return 1.0 / std::sqrt(input);
  }
  return std::sqrt(input);
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

bool IsExecutableSeedOpcode(std::string_view opcode) {
  for (std::string_view supported_opcode : kExecutableSeedOpcodes) {
    if (supported_opcode == opcode) {
      return true;
    }
  }
  return false;
}

std::string_view NormalizeExecutableSeedOpcode(std::string_view opcode) {
  if (opcode == "S_ADD_CO_U32") {
    return "S_ADD_U32";
  }
  if (opcode == "S_ADD_CO_I32") {
    return "S_ADD_I32";
  }
  if (opcode == "S_SUB_CO_U32") {
    return "S_SUB_U32";
  }
  if (opcode == "V_ADD_NC_U32") {
    return "V_ADD_U32";
  }
  if (opcode == "V_SUB_NC_U32") {
    return "V_SUB_U32";
  }
  if (opcode == "V_SUBREV_NC_U32") {
    return "V_SUBREV_U32";
  }
  return opcode;
}

std::string_view ImportedSupportOpcode(std::string_view opcode) {
  const std::string_view normalized_opcode = NormalizeExecutableSeedOpcode(opcode);
  if (normalized_opcode == "S_ADD_U32") {
    return "S_ADD_CO_U32";
  }
  if (normalized_opcode == "S_ADD_I32") {
    return "S_ADD_CO_I32";
  }
  if (normalized_opcode == "S_SUB_U32") {
    return "S_SUB_CO_U32";
  }
  if (normalized_opcode == "V_ADD_U32") {
    return "V_ADD_NC_U32";
  }
  if (normalized_opcode == "V_SUB_U32") {
    return "V_SUB_NC_U32";
  }
  if (normalized_opcode == "V_SUBREV_U32") {
    return "V_SUBREV_NC_U32";
  }
  return normalized_opcode;
}

std::string BuildExecutableOpcodeList() {
  std::string message;
  bool first = true;
  for (std::string_view opcode : kExecutableSeedOpcodes) {
    message.append(first ? "" : ", ");
    message.append(opcode);
    first = false;
  }
  return message;
}

std::string BuildInterpreterBringupMessage(std::string_view missing_opcode = {}) {
  std::string message = "gfx1201 interpreter seed slice supports ";
  message.append(BuildExecutableOpcodeList());

  if (!missing_opcode.empty()) {
    message.append("; ");
    message.append(missing_opcode);
    message.append(" is not in the executable seed slice");
  }

  message.append("; carry-over families:");
  bool first = true;
  for (const Gfx1201FamilyFocus& focus : GetGfx1201CarryOverFamilyFocus()) {
    if (focus.bucket != Gfx1201SupportBucket::kTransferableFull) {
      continue;
    }
    message.append(first ? " " : ", ");
    message.append(focus.family_name);
    first = false;
  }

  message.append("; RDNA4 delta families:");
  first = true;
  for (const Gfx1201FamilyFocus& focus : GetGfx1201Rdna4DeltaFamilyFocus()) {
    message.append(first ? " " : ", ");
    message.append(focus.family_name);
    first = false;
  }
  return message;
}

bool TryCompileExecutableOpcode(std::string_view opcode,
                                Gfx1201CompiledOpcode* compiled_opcode) {
  if (compiled_opcode == nullptr) {
    return false;
  }
  if (opcode == "S_ENDPGM") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSEndpgm;
    return true;
  }
  if (opcode == "S_NOP") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSNop;
    return true;
  }
  if (opcode == "S_ADD_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSAddU32;
    return true;
  }
  if (opcode == "S_ADD_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSAddI32;
    return true;
  }
  if (opcode == "S_SUB_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSSubU32;
    return true;
  }
  if (opcode == "S_CMP_EQ_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSCmpEqI32;
    return true;
  }
  if (opcode == "S_CMP_LG_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSCmpLgI32;
    return true;
  }
  if (opcode == "S_CMP_GT_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSCmpGtI32;
    return true;
  }
  if (opcode == "S_CMP_EQ_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSCmpEqU32;
    return true;
  }
  if (opcode == "S_CMP_LG_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSCmpLgU32;
    return true;
  }
  if (opcode == "S_CMP_GE_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSCmpGeI32;
    return true;
  }
  if (opcode == "S_CMP_LT_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSCmpLtI32;
    return true;
  }
  if (opcode == "S_CMP_LE_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSCmpLeI32;
    return true;
  }
  if (opcode == "S_CMP_GT_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSCmpGtU32;
    return true;
  }
  if (opcode == "S_CMP_GE_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSCmpGeU32;
    return true;
  }
  if (opcode == "S_CMP_LT_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSCmpLtU32;
    return true;
  }
  if (opcode == "S_CMP_LE_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSCmpLeU32;
    return true;
  }
  if (opcode == "S_BRANCH") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSBranch;
    return true;
  }
  if (opcode == "S_CBRANCH_SCC0") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSCbranchScc0;
    return true;
  }
  if (opcode == "S_CBRANCH_SCC1") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSCbranchScc1;
    return true;
  }
  if (opcode == "S_CBRANCH_VCCZ") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSCbranchVccz;
    return true;
  }
  if (opcode == "S_CBRANCH_VCCNZ") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSCbranchVccnz;
    return true;
  }
  if (opcode == "S_CBRANCH_EXECZ") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSCbranchExecz;
    return true;
  }
  if (opcode == "S_CBRANCH_EXECNZ") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSCbranchExecnz;
    return true;
  }
  if (opcode == "S_MOV_B32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSMovB32;
    return true;
  }
  if (opcode == "S_MOVK_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kSMovkI32;
    return true;
  }
  if (opcode == "V_MOV_B32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVMovB32;
    return true;
  }
  if (opcode == "V_CMP_EQ_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpEqI32;
    return true;
  }
  if (opcode == "V_CMP_NE_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpNeI32;
    return true;
  }
  if (opcode == "V_CMP_LT_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpLtI32;
    return true;
  }
  if (opcode == "V_CMP_LE_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpLeI32;
    return true;
  }
  if (opcode == "V_CMP_GT_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpGtI32;
    return true;
  }
  if (opcode == "V_CMP_GE_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpGeI32;
    return true;
  }
  if (opcode == "V_CMP_EQ_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpEqU32;
    return true;
  }
  if (opcode == "V_CMP_NE_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpNeU32;
    return true;
  }
  if (opcode == "V_CMP_LT_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpLtU32;
    return true;
  }
  if (opcode == "V_CMP_LE_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpLeU32;
    return true;
  }
  if (opcode == "V_CMP_GT_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpGtU32;
    return true;
  }
  if (opcode == "V_CMP_GE_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpGeU32;
    return true;
  }
  if (opcode == "V_CMPX_EQ_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxEqI32;
    return true;
  }
  if (opcode == "V_CMPX_NE_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxNeI32;
    return true;
  }
  if (opcode == "V_CMPX_LT_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxLtI32;
    return true;
  }
  if (opcode == "V_CMPX_LE_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxLeI32;
    return true;
  }
  if (opcode == "V_CMPX_GT_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxGtI32;
    return true;
  }
  if (opcode == "V_CMPX_GE_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxGeI32;
    return true;
  }
  if (opcode == "V_CMPX_EQ_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxEqU32;
    return true;
  }
  if (opcode == "V_CMPX_NE_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxNeU32;
    return true;
  }
  if (opcode == "V_CMPX_LT_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxLtU32;
    return true;
  }
  if (opcode == "V_CMPX_LE_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxLeU32;
    return true;
  }
  if (opcode == "V_CMPX_GT_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxGtU32;
    return true;
  }
  if (opcode == "V_CMPX_GE_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxGeU32;
    return true;
  }
  if (opcode == "V_CMP_EQ_I64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpEqI64;
    return true;
  }
  if (opcode == "V_CMP_NE_I64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpNeI64;
    return true;
  }
  if (opcode == "V_CMP_LT_I64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpLtI64;
    return true;
  }
  if (opcode == "V_CMP_LE_I64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpLeI64;
    return true;
  }
  if (opcode == "V_CMP_GT_I64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpGtI64;
    return true;
  }
  if (opcode == "V_CMP_GE_I64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpGeI64;
    return true;
  }
  if (opcode == "V_CMP_EQ_U64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpEqU64;
    return true;
  }
  if (opcode == "V_CMP_NE_U64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpNeU64;
    return true;
  }
  if (opcode == "V_CMP_LT_U64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpLtU64;
    return true;
  }
  if (opcode == "V_CMP_LE_U64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpLeU64;
    return true;
  }
  if (opcode == "V_CMP_GT_U64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpGtU64;
    return true;
  }
  if (opcode == "V_CMP_GE_U64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpGeU64;
    return true;
  }
  if (opcode == "V_CMPX_EQ_I64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxEqI64;
    return true;
  }
  if (opcode == "V_CMPX_NE_I64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxNeI64;
    return true;
  }
  if (opcode == "V_CMPX_LT_I64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxLtI64;
    return true;
  }
  if (opcode == "V_CMPX_LE_I64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxLeI64;
    return true;
  }
  if (opcode == "V_CMPX_GT_I64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxGtI64;
    return true;
  }
  if (opcode == "V_CMPX_GE_I64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxGeI64;
    return true;
  }
  if (opcode == "V_CMPX_EQ_U64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxEqU64;
    return true;
  }
  if (opcode == "V_CMPX_NE_U64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxNeU64;
    return true;
  }
  if (opcode == "V_CMPX_LT_U64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxLtU64;
    return true;
  }
  if (opcode == "V_CMPX_LE_U64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxLeU64;
    return true;
  }
  if (opcode == "V_CMPX_GT_U64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxGtU64;
    return true;
  }
  if (opcode == "V_CMPX_GE_U64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxGeU64;
    return true;
  }
  if (opcode == "V_CMP_CLASS_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpClassF32;
    return true;
  }
  if (opcode == "V_CMP_EQ_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpEqF32;
    return true;
  }
  if (opcode == "V_CMP_GE_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpGeF32;
    return true;
  }
  if (opcode == "V_CMP_GT_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpGtF32;
    return true;
  }
  if (opcode == "V_CMP_LE_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpLeF32;
    return true;
  }
  if (opcode == "V_CMP_LG_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpLgF32;
    return true;
  }
  if (opcode == "V_CMP_LT_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpLtF32;
    return true;
  }
  if (opcode == "V_CMP_NEQ_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpNeqF32;
    return true;
  }
  if (opcode == "V_CMP_O_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpOF32;
    return true;
  }
  if (opcode == "V_CMP_U_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpUF32;
    return true;
  }
  if (opcode == "V_CMP_NGE_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpNgeF32;
    return true;
  }
  if (opcode == "V_CMP_NLG_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpNlgF32;
    return true;
  }
  if (opcode == "V_CMP_NGT_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpNgtF32;
    return true;
  }
  if (opcode == "V_CMP_NLE_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpNleF32;
    return true;
  }
  if (opcode == "V_CMP_NLT_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpNltF32;
    return true;
  }
  if (opcode == "V_CMPX_CLASS_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxClassF32;
    return true;
  }
  if (opcode == "V_CMPX_EQ_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxEqF32;
    return true;
  }
  if (opcode == "V_CMPX_GE_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxGeF32;
    return true;
  }
  if (opcode == "V_CMPX_GT_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxGtF32;
    return true;
  }
  if (opcode == "V_CMPX_LE_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxLeF32;
    return true;
  }
  if (opcode == "V_CMPX_LG_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxLgF32;
    return true;
  }
  if (opcode == "V_CMPX_LT_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxLtF32;
    return true;
  }
  if (opcode == "V_CMPX_NEQ_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxNeqF32;
    return true;
  }
  if (opcode == "V_CMPX_O_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxOF32;
    return true;
  }
  if (opcode == "V_CMPX_U_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxUF32;
    return true;
  }
  if (opcode == "V_CMPX_NGE_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxNgeF32;
    return true;
  }
  if (opcode == "V_CMPX_NLG_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxNlgF32;
    return true;
  }
  if (opcode == "V_CMPX_NGT_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxNgtF32;
    return true;
  }
  if (opcode == "V_CMPX_NLE_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxNleF32;
    return true;
  }
  if (opcode == "V_CMPX_NLT_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxNltF32;
    return true;
  }
  if (opcode == "V_CMP_CLASS_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpClassF64;
    return true;
  }
  if (opcode == "V_CMP_EQ_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpEqF64;
    return true;
  }
  if (opcode == "V_CMP_GE_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpGeF64;
    return true;
  }
  if (opcode == "V_CMP_GT_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpGtF64;
    return true;
  }
  if (opcode == "V_CMP_LE_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpLeF64;
    return true;
  }
  if (opcode == "V_CMP_LG_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpLgF64;
    return true;
  }
  if (opcode == "V_CMP_LT_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpLtF64;
    return true;
  }
  if (opcode == "V_CMP_NEQ_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpNeqF64;
    return true;
  }
  if (opcode == "V_CMP_O_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpOF64;
    return true;
  }
  if (opcode == "V_CMP_U_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpUF64;
    return true;
  }
  if (opcode == "V_CMP_NGE_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpNgeF64;
    return true;
  }
  if (opcode == "V_CMP_NLG_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpNlgF64;
    return true;
  }
  if (opcode == "V_CMP_NGT_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpNgtF64;
    return true;
  }
  if (opcode == "V_CMP_NLE_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpNleF64;
    return true;
  }
  if (opcode == "V_CMP_NLT_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpNltF64;
    return true;
  }
  if (opcode == "V_CMPX_CLASS_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxClassF64;
    return true;
  }
  if (opcode == "V_CMPX_EQ_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxEqF64;
    return true;
  }
  if (opcode == "V_CMPX_GE_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxGeF64;
    return true;
  }
  if (opcode == "V_CMPX_GT_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxGtF64;
    return true;
  }
  if (opcode == "V_CMPX_LE_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxLeF64;
    return true;
  }
  if (opcode == "V_CMPX_LG_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxLgF64;
    return true;
  }
  if (opcode == "V_CMPX_LT_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxLtF64;
    return true;
  }
  if (opcode == "V_CMPX_NEQ_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxNeqF64;
    return true;
  }
  if (opcode == "V_CMPX_O_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxOF64;
    return true;
  }
  if (opcode == "V_CMPX_U_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxUF64;
    return true;
  }
  if (opcode == "V_CMPX_NGE_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxNgeF64;
    return true;
  }
  if (opcode == "V_CMPX_NLG_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxNlgF64;
    return true;
  }
  if (opcode == "V_CMPX_NGT_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxNgtF64;
    return true;
  }
  if (opcode == "V_CMPX_NLE_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxNleF64;
    return true;
  }
  if (opcode == "V_CMPX_NLT_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCmpxNltF64;
    return true;
  }
  if (opcode == "V_NOT_B32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVNotB32;
    return true;
  }
  if (opcode == "V_BFREV_B32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVBfrevB32;
    return true;
  }
  if (opcode == "V_CLS_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVClsI32;
    return true;
  }
  if (opcode == "V_CLZ_I32_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVClzI32U32;
    return true;
  }
  if (opcode == "V_CTZ_I32_B32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCtzI32B32;
    return true;
  }
  if (opcode == "V_CVT_F32_UBYTE0") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCvtF32Ubyte0;
    return true;
  }
  if (opcode == "V_CVT_F32_UBYTE1") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCvtF32Ubyte1;
    return true;
  }
  if (opcode == "V_CVT_F32_UBYTE2") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCvtF32Ubyte2;
    return true;
  }
  if (opcode == "V_CVT_F32_UBYTE3") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCvtF32Ubyte3;
    return true;
  }
  if (opcode == "V_CVT_F32_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCvtF32I32;
    return true;
  }
  if (opcode == "V_CVT_F32_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCvtF32U32;
    return true;
  }
  if (opcode == "V_CVT_F32_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCvtF32F64;
    return true;
  }
  if (opcode == "V_CVT_F64_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCvtF64F32;
    return true;
  }
  if (opcode == "V_CVT_F64_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCvtF64I32;
    return true;
  }
  if (opcode == "V_CVT_F64_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCvtF64U32;
    return true;
  }
  if (opcode == "V_CVT_U32_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCvtU32F32;
    return true;
  }
  if (opcode == "V_CVT_U32_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCvtU32F64;
    return true;
  }
  if (opcode == "V_CVT_I32_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCvtI32F32;
    return true;
  }
  if (opcode == "V_CVT_FLOOR_I32_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCvtFloorI32F32;
    return true;
  }
  if (opcode == "V_CVT_NEAREST_I32_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCvtNearestI32F32;
    return true;
  }
  if (opcode == "V_CVT_I32_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCvtI32F64;
    return true;
  }
  if (opcode == "V_EXP_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVExpF32;
    return true;
  }
  if (opcode == "V_LOG_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVLogF32;
    return true;
  }
  if (opcode == "V_RCP_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVRcpF32;
    return true;
  }
  if (opcode == "V_RCP_IFLAG_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVRcpIflagF32;
    return true;
  }
  if (opcode == "V_RSQ_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVRsqF32;
    return true;
  }
  if (opcode == "V_SQRT_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVSqrtF32;
    return true;
  }
  if (opcode == "V_SIN_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVSinF32;
    return true;
  }
  if (opcode == "V_COS_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCosF32;
    return true;
  }
  if (opcode == "V_RCP_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVRcpF64;
    return true;
  }
  if (opcode == "V_RSQ_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVRsqF64;
    return true;
  }
  if (opcode == "V_SQRT_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVSqrtF64;
    return true;
  }
  if (opcode == "V_FREXP_EXP_I32_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVFrexpExpI32F32;
    return true;
  }
  if (opcode == "V_FREXP_MANT_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVFrexpMantF32;
    return true;
  }
  if (opcode == "V_FRACT_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVFractF32;
    return true;
  }
  if (opcode == "V_FREXP_EXP_I32_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVFrexpExpI32F64;
    return true;
  }
  if (opcode == "V_FREXP_MANT_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVFrexpMantF64;
    return true;
  }
  if (opcode == "V_FRACT_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVFractF64;
    return true;
  }
  if (opcode == "V_TRUNC_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVTruncF32;
    return true;
  }
  if (opcode == "V_CEIL_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCeilF32;
    return true;
  }
  if (opcode == "V_RNDNE_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVRndneF32;
    return true;
  }
  if (opcode == "V_FLOOR_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVFloorF32;
    return true;
  }
  if (opcode == "V_TRUNC_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVTruncF64;
    return true;
  }
  if (opcode == "V_CEIL_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCeilF64;
    return true;
  }
  if (opcode == "V_RNDNE_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVRndneF64;
    return true;
  }
  if (opcode == "V_FLOOR_F64") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVFloorF64;
    return true;
  }
  if (opcode == "V_ADD_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVAddU32;
    return true;
  }
  if (opcode == "V_SUB_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVSubU32;
    return true;
  }
  if (opcode == "V_SUBREV_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVSubrevU32;
    return true;
  }
  if (opcode == "V_MIN_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVMinI32;
    return true;
  }
  if (opcode == "V_MAX_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVMaxI32;
    return true;
  }
  if (opcode == "V_MIN_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVMinU32;
    return true;
  }
  if (opcode == "V_MAX_U32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVMaxU32;
    return true;
  }
  if (opcode == "V_CNDMASK_B32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCndmaskB32;
    return true;
  }
  if (opcode == "V_LSHRREV_B32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVLshrrevB32;
    return true;
  }
  if (opcode == "V_ASHRREV_I32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVAshrrevI32;
    return true;
  }
  if (opcode == "V_LSHLREV_B32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVLshlrevB32;
    return true;
  }
  if (opcode == "V_AND_B32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVAndB32;
    return true;
  }
  if (opcode == "V_OR_B32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVOrB32;
    return true;
  }
  if (opcode == "V_XOR_B32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVXorB32;
    return true;
  }
  *compiled_opcode = Gfx1201CompiledOpcode::kUnknown;
  return false;
}

bool ApplyRelativeBranch(std::int32_t delta_in_instructions,
                         WaveExecutionState* state,
                         bool* pc_was_updated,
                         std::string* error_message) {
  const std::int64_t target_pc =
      static_cast<std::int64_t>(state->pc) + 1 +
      static_cast<std::int64_t>(delta_in_instructions);
  if (target_pc < 0) {
    if (error_message != nullptr) {
      *error_message = "branch target underflow";
    }
    return false;
  }

  state->pc = static_cast<std::uint64_t>(target_pc);
  if (pc_was_updated != nullptr) {
    *pc_was_updated = true;
  }
  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool ValidateOperandCount(const DecodedInstruction& instruction,
                          std::uint8_t expected_operand_count,
                          std::string* error_message) {
  if (instruction.operand_count != expected_operand_count) {
    if (error_message != nullptr) {
      *error_message = "unexpected operand count";
    }
    return false;
  }
  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool ValidateOperandCount(const Gfx1201CompiledInstruction& instruction,
                          std::uint8_t expected_operand_count,
                          std::string* error_message) {
  return ValidateOperandCount(instruction.decoded_instruction,
                              expected_operand_count, error_message);
}

std::uint32_t ReadScalarOperand(const InstructionOperand& operand,
                                const WaveExecutionState& state,
                                std::string* error_message) {
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
    if (operand.index == kImplicitVccPairSgprIndex) {
      return static_cast<std::uint32_t>(state.vcc_mask);
    }
    if (operand.index == kImplicitVccPairSgprIndex + 1) {
      return static_cast<std::uint32_t>(state.vcc_mask >> 32);
    }
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

std::uint32_t ReadVectorOperand(const InstructionOperand& operand,
                                const WaveExecutionState& state,
                                std::size_t lane_index,
                                std::string* error_message) {
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

std::uint64_t ReadWideSourceOperand(const InstructionOperand& operand,
                                    const WaveExecutionState& state,
                                    std::size_t lane_index,
                                    std::string* error_message) {
  if (error_message != nullptr) {
    error_message->clear();
  }

  if (operand.kind == OperandKind::kImm32) {
    return operand.imm32;
  }
  if (operand.kind == OperandKind::kVgpr) {
    if (operand.index + 1 >= state.vgprs.size()) {
      if (error_message != nullptr) {
        *error_message = "vector register pair out of range";
      }
      return 0;
    }
    return static_cast<std::uint64_t>(state.vgprs[operand.index][lane_index]) |
           (static_cast<std::uint64_t>(
                state.vgprs[operand.index + 1][lane_index])
            << 32);
  }
  if (operand.kind != OperandKind::kSgpr) {
    if (error_message != nullptr) {
      *error_message = "unsupported wide operand kind";
    }
    return 0;
  }

  if (operand.index == kExecPairSgprIndex) {
    return state.exec_mask;
  }

  if (operand.index >= state.sgprs.size()) {
    if (operand.index == kImplicitVccPairSgprIndex) {
      return state.vcc_mask;
    }
    if (error_message != nullptr) {
      *error_message = "scalar register pair out of range";
    }
    return 0;
  }
  if (operand.index + 1 >= state.sgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "scalar register pair out of range";
    }
    return 0;
  }
  return static_cast<std::uint64_t>(state.sgprs[operand.index]) |
         (static_cast<std::uint64_t>(state.sgprs[operand.index + 1]) << 32);
}

bool WriteScalarOperand(const InstructionOperand& operand,
                        std::uint32_t value,
                        WaveExecutionState* state,
                        std::string* error_message) {
  if (operand.kind != OperandKind::kSgpr) {
    if (error_message != nullptr) {
      *error_message = "expected scalar destination operand";
    }
    return false;
  }
  if (operand.index == kImplicitVccPairSgprIndex) {
    state->vcc_mask = (state->vcc_mask & 0xffffffff00000000ULL) | value;
    state->ClampMasksToLaneCount();
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }
  if (operand.index == kImplicitVccPairSgprIndex + 1) {
    state->vcc_mask = (state->vcc_mask & 0x00000000ffffffffULL) |
                      (static_cast<std::uint64_t>(value) << 32);
    state->ClampMasksToLaneCount();
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }
  if (operand.index >= state->sgprs.size()) {
    if (error_message != nullptr) {
      *error_message = "scalar destination out of range";
    }
    return false;
  }

  state->sgprs[operand.index] = value;
  if (operand.index == kExecPairSgprIndex) {
    state->exec_mask = (state->exec_mask & 0xffffffff00000000ULL) | value;
  } else if (operand.index == kExecPairSgprIndex + 1) {
    state->exec_mask = (state->exec_mask & 0x00000000ffffffffULL) |
                       (static_cast<std::uint64_t>(value) << 32);
  }
  state->ClampMasksToLaneCount();

  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool WriteVectorOperand(const InstructionOperand& operand,
                        std::size_t lane_index,
                        std::uint32_t value,
                        WaveExecutionState* state,
                        std::string* error_message) {
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

  state->vgprs[operand.index][lane_index] = value;
  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool WriteWideVectorOperand(const InstructionOperand& operand,
                            std::size_t lane_index,
                            std::uint64_t value,
                            WaveExecutionState* state,
                            std::string* error_message) {
  if (operand.kind != OperandKind::kVgpr) {
    if (error_message != nullptr) {
      *error_message = "expected vector destination operand";
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

std::uint32_t EvaluateVectorUnarySeedInstruction(std::string_view opcode,
                                                 std::uint32_t value) {
  if (opcode == "V_NOT_B32") {
    return ~value;
  }
  if (opcode == "V_BFREV_B32") {
    return ReverseBits32(value);
  }
  if (opcode == "V_CLS_I32") {
    return FindFirstBitHighSigned(value);
  }
  if (opcode == "V_CLZ_I32_U32") {
    return FindFirstBitHighUnsigned(value);
  }
  if (opcode == "V_CTZ_I32_B32") {
    return FindFirstBitLow(value);
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
    return BitCast<std::uint32_t>(
        static_cast<float>(BitCast<std::int32_t>(value)));
  }
  if (opcode == "V_CVT_F32_U32") {
    return BitCast<std::uint32_t>(static_cast<float>(value));
  }
  if (opcode == "V_EXP_F32" || opcode == "V_LOG_F32" ||
      opcode == "V_RCP_F32" || opcode == "V_RCP_IFLAG_F32" ||
      opcode == "V_RSQ_F32" || opcode == "V_SQRT_F32" ||
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
  if (opcode == "V_CVT_U32_F32") {
    return TruncateFloatToU32(BitCast<float>(value));
  }
  if (opcode == "V_CVT_I32_F32") {
    return BitCast<std::uint32_t>(TruncateFloatToI32(BitCast<float>(value)));
  }
  if (opcode == "V_CVT_FLOOR_I32_F32") {
    return BitCast<std::uint32_t>(FloorFloatToI32(BitCast<float>(value)));
  }
  if (opcode == "V_CVT_NEAREST_I32_F32") {
    return BitCast<std::uint32_t>(RoundNearestFloatToI32(BitCast<float>(value)));
  }
  return value;
}

std::uint64_t EvaluateWideVectorUnarySeedInstruction(std::string_view opcode,
                                                     std::uint32_t value) {
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

std::uint32_t EvaluateVectorUnaryFromWideSeedInstruction(std::string_view opcode,
                                                         std::uint64_t value) {
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

std::uint64_t EvaluateWideVectorUnaryToWideSeedInstruction(
    std::string_view opcode, std::uint64_t value) {
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

std::uint32_t EvaluateVectorBinarySeedInstruction(std::string_view opcode,
                                                  std::uint32_t lhs,
                                                  std::uint32_t rhs) {
  if (opcode == "V_ADD_U32") {
    return lhs + rhs;
  }
  if (opcode == "V_SUB_U32") {
    return lhs - rhs;
  }
  if (opcode == "V_SUBREV_U32") {
    return rhs - lhs;
  }
  if (opcode == "V_MIN_I32") {
    return static_cast<std::uint32_t>(
        std::min(BitCast<std::int32_t>(lhs), BitCast<std::int32_t>(rhs)));
  }
  if (opcode == "V_MAX_I32") {
    return static_cast<std::uint32_t>(
        std::max(BitCast<std::int32_t>(lhs), BitCast<std::int32_t>(rhs)));
  }
  if (opcode == "V_MIN_U32") {
    return std::min(lhs, rhs);
  }
  if (opcode == "V_MAX_U32") {
    return std::max(lhs, rhs);
  }
  if (opcode == "V_LSHRREV_B32") {
    return rhs >> (lhs & 31u);
  }
  if (opcode == "V_ASHRREV_I32") {
    return static_cast<std::uint32_t>(BitCast<std::int32_t>(rhs) >> (lhs & 31u));
  }
  if (opcode == "V_LSHLREV_B32") {
    return rhs << (lhs & 31u);
  }
  if (opcode == "V_AND_B32") {
    return lhs & rhs;
  }
  if (opcode == "V_OR_B32") {
    return lhs | rhs;
  }
  if (opcode == "V_XOR_B32") {
    return lhs ^ rhs;
  }
  return 0u;
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
  const std::uint64_t exponent = (bits >> 52) & 0x7ffu;
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

std::string_view NormalizeVectorCompareSeedOpcode(std::string_view opcode) {
  if (opcode == "V_CMPX_EQ_I32") {
    return "V_CMP_EQ_I32";
  }
  if (opcode == "V_CMPX_NE_I32") {
    return "V_CMP_NE_I32";
  }
  if (opcode == "V_CMPX_LT_I32") {
    return "V_CMP_LT_I32";
  }
  if (opcode == "V_CMPX_LE_I32") {
    return "V_CMP_LE_I32";
  }
  if (opcode == "V_CMPX_GT_I32") {
    return "V_CMP_GT_I32";
  }
  if (opcode == "V_CMPX_GE_I32") {
    return "V_CMP_GE_I32";
  }
  if (opcode == "V_CMPX_EQ_U32") {
    return "V_CMP_EQ_U32";
  }
  if (opcode == "V_CMPX_NE_U32") {
    return "V_CMP_NE_U32";
  }
  if (opcode == "V_CMPX_LT_U32") {
    return "V_CMP_LT_U32";
  }
  if (opcode == "V_CMPX_LE_U32") {
    return "V_CMP_LE_U32";
  }
  if (opcode == "V_CMPX_GT_U32") {
    return "V_CMP_GT_U32";
  }
  if (opcode == "V_CMPX_GE_U32") {
    return "V_CMP_GE_U32";
  }
  if (opcode == "V_CMPX_EQ_I64") {
    return "V_CMP_EQ_I64";
  }
  if (opcode == "V_CMPX_NE_I64") {
    return "V_CMP_NE_I64";
  }
  if (opcode == "V_CMPX_LT_I64") {
    return "V_CMP_LT_I64";
  }
  if (opcode == "V_CMPX_LE_I64") {
    return "V_CMP_LE_I64";
  }
  if (opcode == "V_CMPX_GT_I64") {
    return "V_CMP_GT_I64";
  }
  if (opcode == "V_CMPX_GE_I64") {
    return "V_CMP_GE_I64";
  }
  if (opcode == "V_CMPX_EQ_U64") {
    return "V_CMP_EQ_U64";
  }
  if (opcode == "V_CMPX_NE_U64") {
    return "V_CMP_NE_U64";
  }
  if (opcode == "V_CMPX_LT_U64") {
    return "V_CMP_LT_U64";
  }
  if (opcode == "V_CMPX_LE_U64") {
    return "V_CMP_LE_U64";
  }
  if (opcode == "V_CMPX_GT_U64") {
    return "V_CMP_GT_U64";
  }
  if (opcode == "V_CMPX_GE_U64") {
    return "V_CMP_GE_U64";
  }
  if (opcode == "V_CMPX_CLASS_F32") {
    return "V_CMP_CLASS_F32";
  }
  if (opcode == "V_CMPX_EQ_F32") {
    return "V_CMP_EQ_F32";
  }
  if (opcode == "V_CMPX_GE_F32") {
    return "V_CMP_GE_F32";
  }
  if (opcode == "V_CMPX_GT_F32") {
    return "V_CMP_GT_F32";
  }
  if (opcode == "V_CMPX_LE_F32") {
    return "V_CMP_LE_F32";
  }
  if (opcode == "V_CMPX_LG_F32") {
    return "V_CMP_LG_F32";
  }
  if (opcode == "V_CMPX_LT_F32") {
    return "V_CMP_LT_F32";
  }
  if (opcode == "V_CMPX_NEQ_F32") {
    return "V_CMP_NEQ_F32";
  }
  if (opcode == "V_CMPX_O_F32") {
    return "V_CMP_O_F32";
  }
  if (opcode == "V_CMPX_U_F32") {
    return "V_CMP_U_F32";
  }
  if (opcode == "V_CMPX_NGE_F32") {
    return "V_CMP_NGE_F32";
  }
  if (opcode == "V_CMPX_NLG_F32") {
    return "V_CMP_NLG_F32";
  }
  if (opcode == "V_CMPX_NGT_F32") {
    return "V_CMP_NGT_F32";
  }
  if (opcode == "V_CMPX_NLE_F32") {
    return "V_CMP_NLE_F32";
  }
  if (opcode == "V_CMPX_NLT_F32") {
    return "V_CMP_NLT_F32";
  }
  if (opcode == "V_CMPX_CLASS_F64") {
    return "V_CMP_CLASS_F64";
  }
  if (opcode == "V_CMPX_EQ_F64") {
    return "V_CMP_EQ_F64";
  }
  if (opcode == "V_CMPX_GE_F64") {
    return "V_CMP_GE_F64";
  }
  if (opcode == "V_CMPX_GT_F64") {
    return "V_CMP_GT_F64";
  }
  if (opcode == "V_CMPX_LE_F64") {
    return "V_CMP_LE_F64";
  }
  if (opcode == "V_CMPX_LG_F64") {
    return "V_CMP_LG_F64";
  }
  if (opcode == "V_CMPX_LT_F64") {
    return "V_CMP_LT_F64";
  }
  if (opcode == "V_CMPX_NEQ_F64") {
    return "V_CMP_NEQ_F64";
  }
  if (opcode == "V_CMPX_O_F64") {
    return "V_CMP_O_F64";
  }
  if (opcode == "V_CMPX_U_F64") {
    return "V_CMP_U_F64";
  }
  if (opcode == "V_CMPX_NGE_F64") {
    return "V_CMP_NGE_F64";
  }
  if (opcode == "V_CMPX_NLG_F64") {
    return "V_CMP_NLG_F64";
  }
  if (opcode == "V_CMPX_NGT_F64") {
    return "V_CMP_NGT_F64";
  }
  if (opcode == "V_CMPX_NLE_F64") {
    return "V_CMP_NLE_F64";
  }
  if (opcode == "V_CMPX_NLT_F64") {
    return "V_CMP_NLT_F64";
  }
  return opcode;
}

bool IsVectorCmpxSeedInstruction(std::string_view opcode) {
  return NormalizeVectorCompareSeedOpcode(opcode) != opcode;
}

bool IsVectorCompareSeedInstruction(std::string_view opcode) {
  const std::string_view normalized_opcode =
      NormalizeVectorCompareSeedOpcode(opcode);
  return normalized_opcode == "V_CMP_EQ_I32" ||
         normalized_opcode == "V_CMP_NE_I32" ||
         normalized_opcode == "V_CMP_LT_I32" ||
         normalized_opcode == "V_CMP_LE_I32" ||
         normalized_opcode == "V_CMP_GT_I32" ||
         normalized_opcode == "V_CMP_GE_I32" ||
         normalized_opcode == "V_CMP_EQ_U32" ||
         normalized_opcode == "V_CMP_NE_U32" ||
         normalized_opcode == "V_CMP_LT_U32" ||
         normalized_opcode == "V_CMP_LE_U32" ||
         normalized_opcode == "V_CMP_GT_U32" ||
         normalized_opcode == "V_CMP_GE_U32" ||
         normalized_opcode == "V_CMP_EQ_I64" ||
         normalized_opcode == "V_CMP_NE_I64" ||
         normalized_opcode == "V_CMP_LT_I64" ||
         normalized_opcode == "V_CMP_LE_I64" ||
         normalized_opcode == "V_CMP_GT_I64" ||
         normalized_opcode == "V_CMP_GE_I64" ||
         normalized_opcode == "V_CMP_EQ_U64" ||
         normalized_opcode == "V_CMP_NE_U64" ||
         normalized_opcode == "V_CMP_LT_U64" ||
         normalized_opcode == "V_CMP_LE_U64" ||
         normalized_opcode == "V_CMP_GT_U64" ||
         normalized_opcode == "V_CMP_GE_U64" ||
         normalized_opcode == "V_CMP_CLASS_F32" ||
         normalized_opcode == "V_CMP_EQ_F32" ||
         normalized_opcode == "V_CMP_GE_F32" ||
         normalized_opcode == "V_CMP_GT_F32" ||
         normalized_opcode == "V_CMP_LE_F32" ||
         normalized_opcode == "V_CMP_LG_F32" ||
         normalized_opcode == "V_CMP_LT_F32" ||
         normalized_opcode == "V_CMP_NEQ_F32" ||
         normalized_opcode == "V_CMP_O_F32" ||
         normalized_opcode == "V_CMP_U_F32" ||
         normalized_opcode == "V_CMP_NGE_F32" ||
         normalized_opcode == "V_CMP_NLG_F32" ||
         normalized_opcode == "V_CMP_NGT_F32" ||
         normalized_opcode == "V_CMP_NLE_F32" ||
         normalized_opcode == "V_CMP_NLT_F32" ||
         normalized_opcode == "V_CMP_CLASS_F64" ||
         normalized_opcode == "V_CMP_EQ_F64" ||
         normalized_opcode == "V_CMP_GE_F64" ||
         normalized_opcode == "V_CMP_GT_F64" ||
         normalized_opcode == "V_CMP_LE_F64" ||
         normalized_opcode == "V_CMP_LG_F64" ||
         normalized_opcode == "V_CMP_LT_F64" ||
         normalized_opcode == "V_CMP_NEQ_F64" ||
         normalized_opcode == "V_CMP_O_F64" ||
         normalized_opcode == "V_CMP_U_F64" ||
         normalized_opcode == "V_CMP_NGE_F64" ||
         normalized_opcode == "V_CMP_NLG_F64" ||
         normalized_opcode == "V_CMP_NGT_F64" ||
         normalized_opcode == "V_CMP_NLE_F64" ||
         normalized_opcode == "V_CMP_NLT_F64";
}

bool IsWideVectorCompareClassSeedInstruction(std::string_view opcode) {
  return NormalizeVectorCompareSeedOpcode(opcode) == "V_CMP_CLASS_F64";
}

bool IsWideVectorCompareSeedInstruction(std::string_view opcode) {
  const std::string_view normalized_opcode =
      NormalizeVectorCompareSeedOpcode(opcode);
  return normalized_opcode == "V_CMP_EQ_I64" ||
         normalized_opcode == "V_CMP_NE_I64" ||
         normalized_opcode == "V_CMP_LT_I64" ||
         normalized_opcode == "V_CMP_LE_I64" ||
         normalized_opcode == "V_CMP_GT_I64" ||
         normalized_opcode == "V_CMP_GE_I64" ||
         normalized_opcode == "V_CMP_EQ_U64" ||
         normalized_opcode == "V_CMP_NE_U64" ||
         normalized_opcode == "V_CMP_LT_U64" ||
         normalized_opcode == "V_CMP_LE_U64" ||
         normalized_opcode == "V_CMP_GT_U64" ||
         normalized_opcode == "V_CMP_GE_U64" ||
         normalized_opcode == "V_CMP_EQ_F64" ||
         normalized_opcode == "V_CMP_GE_F64" ||
         normalized_opcode == "V_CMP_GT_F64" ||
         normalized_opcode == "V_CMP_LE_F64" ||
         normalized_opcode == "V_CMP_LG_F64" ||
         normalized_opcode == "V_CMP_LT_F64" ||
         normalized_opcode == "V_CMP_NEQ_F64" ||
         normalized_opcode == "V_CMP_O_F64" ||
         normalized_opcode == "V_CMP_U_F64" ||
         normalized_opcode == "V_CMP_NGE_F64" ||
         normalized_opcode == "V_CMP_NLG_F64" ||
         normalized_opcode == "V_CMP_NGT_F64" ||
         normalized_opcode == "V_CMP_NLE_F64" ||
         normalized_opcode == "V_CMP_NLT_F64";
}

bool EvaluateVectorCompareSeedInstruction(std::string_view opcode,
                                          std::uint32_t lhs,
                                          std::uint32_t rhs) {
  const std::string_view normalized_opcode =
      NormalizeVectorCompareSeedOpcode(opcode);
  if (normalized_opcode == "V_CMP_CLASS_F32") {
    return (ClassifyFp32Mask(lhs) & rhs) != 0u;
  }
  if (normalized_opcode == "V_CMP_EQ_F32" ||
      normalized_opcode == "V_CMP_GE_F32" ||
      normalized_opcode == "V_CMP_GT_F32" ||
      normalized_opcode == "V_CMP_LE_F32" ||
      normalized_opcode == "V_CMP_LG_F32" ||
      normalized_opcode == "V_CMP_LT_F32" ||
      normalized_opcode == "V_CMP_NEQ_F32" ||
      normalized_opcode == "V_CMP_O_F32" ||
      normalized_opcode == "V_CMP_U_F32" ||
      normalized_opcode == "V_CMP_NGE_F32" ||
      normalized_opcode == "V_CMP_NLG_F32" ||
      normalized_opcode == "V_CMP_NGT_F32" ||
      normalized_opcode == "V_CMP_NLE_F32" ||
      normalized_opcode == "V_CMP_NLT_F32") {
    const float lhs_float = BitCast<float>(lhs);
    const float rhs_float = BitCast<float>(rhs);
    const bool unordered = std::isnan(lhs_float) || std::isnan(rhs_float);
    if (normalized_opcode == "V_CMP_EQ_F32") {
      return !unordered && lhs_float == rhs_float;
    }
    if (normalized_opcode == "V_CMP_GE_F32") {
      return !unordered && lhs_float >= rhs_float;
    }
    if (normalized_opcode == "V_CMP_GT_F32") {
      return !unordered && lhs_float > rhs_float;
    }
    if (normalized_opcode == "V_CMP_LE_F32") {
      return !unordered && lhs_float <= rhs_float;
    }
    if (normalized_opcode == "V_CMP_LG_F32") {
      return !unordered && lhs_float != rhs_float;
    }
    if (normalized_opcode == "V_CMP_LT_F32") {
      return !unordered && lhs_float < rhs_float;
    }
    if (normalized_opcode == "V_CMP_NEQ_F32") {
      return unordered || lhs_float != rhs_float;
    }
    if (normalized_opcode == "V_CMP_O_F32") {
      return !unordered;
    }
    if (normalized_opcode == "V_CMP_U_F32") {
      return unordered;
    }
    if (normalized_opcode == "V_CMP_NGE_F32") {
      return unordered || lhs_float < rhs_float;
    }
    if (normalized_opcode == "V_CMP_NLG_F32") {
      return unordered || lhs_float == rhs_float;
    }
    if (normalized_opcode == "V_CMP_NGT_F32") {
      return unordered || lhs_float <= rhs_float;
    }
    if (normalized_opcode == "V_CMP_NLE_F32") {
      return unordered || lhs_float > rhs_float;
    }
    return unordered || lhs_float >= rhs_float;
  }
  if (normalized_opcode == "V_CMP_EQ_I32") {
    return BitCast<std::int32_t>(lhs) == BitCast<std::int32_t>(rhs);
  }
  if (normalized_opcode == "V_CMP_NE_I32") {
    return BitCast<std::int32_t>(lhs) != BitCast<std::int32_t>(rhs);
  }
  if (normalized_opcode == "V_CMP_LT_I32") {
    return BitCast<std::int32_t>(lhs) < BitCast<std::int32_t>(rhs);
  }
  if (normalized_opcode == "V_CMP_LE_I32") {
    return BitCast<std::int32_t>(lhs) <= BitCast<std::int32_t>(rhs);
  }
  if (normalized_opcode == "V_CMP_GT_I32") {
    return BitCast<std::int32_t>(lhs) > BitCast<std::int32_t>(rhs);
  }
  if (normalized_opcode == "V_CMP_GE_I32") {
    return BitCast<std::int32_t>(lhs) >= BitCast<std::int32_t>(rhs);
  }
  if (normalized_opcode == "V_CMP_EQ_U32") {
    return lhs == rhs;
  }
  if (normalized_opcode == "V_CMP_NE_U32") {
    return lhs != rhs;
  }
  if (normalized_opcode == "V_CMP_LT_U32") {
    return lhs < rhs;
  }
  if (normalized_opcode == "V_CMP_LE_U32") {
    return lhs <= rhs;
  }
  if (normalized_opcode == "V_CMP_GT_U32") {
    return lhs > rhs;
  }
  if (normalized_opcode == "V_CMP_GE_U32") {
    return lhs >= rhs;
  }
  return false;
}

bool EvaluateWideVectorCompareClassSeedInstruction(std::string_view opcode,
                                                   std::uint64_t lhs,
                                                   std::uint32_t rhs) {
  return NormalizeVectorCompareSeedOpcode(opcode) == "V_CMP_CLASS_F64" &&
         (ClassifyFp64Mask(lhs) & rhs) != 0u;
}

bool EvaluateWideVectorCompareSeedInstruction(std::string_view opcode,
                                              std::uint64_t lhs,
                                              std::uint64_t rhs) {
  const std::string_view normalized_opcode =
      NormalizeVectorCompareSeedOpcode(opcode);
  if (normalized_opcode == "V_CMP_EQ_I64") {
    return BitCast<std::int64_t>(lhs) == BitCast<std::int64_t>(rhs);
  }
  if (normalized_opcode == "V_CMP_NE_I64") {
    return BitCast<std::int64_t>(lhs) != BitCast<std::int64_t>(rhs);
  }
  if (normalized_opcode == "V_CMP_LT_I64") {
    return BitCast<std::int64_t>(lhs) < BitCast<std::int64_t>(rhs);
  }
  if (normalized_opcode == "V_CMP_LE_I64") {
    return BitCast<std::int64_t>(lhs) <= BitCast<std::int64_t>(rhs);
  }
  if (normalized_opcode == "V_CMP_GT_I64") {
    return BitCast<std::int64_t>(lhs) > BitCast<std::int64_t>(rhs);
  }
  if (normalized_opcode == "V_CMP_GE_I64") {
    return BitCast<std::int64_t>(lhs) >= BitCast<std::int64_t>(rhs);
  }
  if (normalized_opcode == "V_CMP_EQ_U64") {
    return lhs == rhs;
  }
  if (normalized_opcode == "V_CMP_NE_U64") {
    return lhs != rhs;
  }
  if (normalized_opcode == "V_CMP_LT_U64") {
    return lhs < rhs;
  }
  if (normalized_opcode == "V_CMP_LE_U64") {
    return lhs <= rhs;
  }
  if (normalized_opcode == "V_CMP_GT_U64") {
    return lhs > rhs;
  }
  if (normalized_opcode == "V_CMP_GE_U64") {
    return lhs >= rhs;
  }
  const double lhs_double = BitCast<double>(lhs);
  const double rhs_double = BitCast<double>(rhs);
  const bool unordered = std::isnan(lhs_double) || std::isnan(rhs_double);
  if (normalized_opcode == "V_CMP_EQ_F64") {
    return !unordered && lhs_double == rhs_double;
  }
  if (normalized_opcode == "V_CMP_GE_F64") {
    return !unordered && lhs_double >= rhs_double;
  }
  if (normalized_opcode == "V_CMP_GT_F64") {
    return !unordered && lhs_double > rhs_double;
  }
  if (normalized_opcode == "V_CMP_LE_F64") {
    return !unordered && lhs_double <= rhs_double;
  }
  if (normalized_opcode == "V_CMP_LG_F64") {
    return !unordered && lhs_double != rhs_double;
  }
  if (normalized_opcode == "V_CMP_LT_F64") {
    return !unordered && lhs_double < rhs_double;
  }
  if (normalized_opcode == "V_CMP_NEQ_F64") {
    return unordered || lhs_double != rhs_double;
  }
  if (normalized_opcode == "V_CMP_O_F64") {
    return !unordered;
  }
  if (normalized_opcode == "V_CMP_U_F64") {
    return unordered;
  }
  if (normalized_opcode == "V_CMP_NGE_F64") {
    return unordered || lhs_double < rhs_double;
  }
  if (normalized_opcode == "V_CMP_NLG_F64") {
    return unordered || lhs_double == rhs_double;
  }
  if (normalized_opcode == "V_CMP_NGT_F64") {
    return unordered || lhs_double <= rhs_double;
  }
  if (normalized_opcode == "V_CMP_NLE_F64") {
    return unordered || lhs_double > rhs_double;
  }
  return normalized_opcode == "V_CMP_NLT_F64" &&
         (unordered || lhs_double >= rhs_double);
}

bool ExecuteDecodedSeedInstruction(const DecodedInstruction& instruction,
                                   WaveExecutionState* state,
                                   bool* pc_was_updated,
                                   std::string* error_message) {
  if (pc_was_updated != nullptr) {
    *pc_was_updated = false;
  }

  if (instruction.opcode == "S_ENDPGM") {
    if (!ValidateOperandCount(instruction, 0, error_message)) {
      return false;
    }
    state->halted = true;
    state->waiting_on_barrier = false;
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }

  if (instruction.opcode == "S_NOP") {
    return ValidateOperandCount(instruction, 1, error_message);
  }

  if (instruction.opcode == "S_BRANCH" || instruction.opcode == "S_CBRANCH_SCC0" ||
      instruction.opcode == "S_CBRANCH_SCC1" ||
      instruction.opcode == "S_CBRANCH_VCCZ" ||
      instruction.opcode == "S_CBRANCH_VCCNZ" ||
      instruction.opcode == "S_CBRANCH_EXECZ" ||
      instruction.opcode == "S_CBRANCH_EXECNZ") {
    if (!ValidateOperandCount(instruction, 1, error_message)) {
      return false;
    }
    if (instruction.operands[0].kind != OperandKind::kImm32) {
      if (error_message != nullptr) {
        *error_message = "expected branch immediate operand";
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
      error_message->clear();
    }
    return true;
  }

  if (instruction.opcode == "S_ADD_U32" || instruction.opcode == "S_ADD_I32" ||
      instruction.opcode == "S_SUB_U32") {
    if (!ValidateOperandCount(instruction, 3, error_message)) {
      return false;
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
    if (instruction.opcode == "S_SUB_U32") {
      result = lhs - rhs;
      state->scc = lhs >= rhs;
    } else {
      const std::uint64_t wide = static_cast<std::uint64_t>(lhs) + rhs;
      result = static_cast<std::uint32_t>(wide);
      state->scc = wide > 0xffffffffULL;
    }
    return WriteScalarOperand(instruction.operands[0], result, state, error_message);
  }

  if (instruction.opcode == "S_CMP_EQ_I32" || instruction.opcode == "S_CMP_LG_I32" ||
      instruction.opcode == "S_CMP_GT_I32" || instruction.opcode == "S_CMP_EQ_U32" ||
      instruction.opcode == "S_CMP_LG_U32" || instruction.opcode == "S_CMP_GE_I32" ||
      instruction.opcode == "S_CMP_LT_I32" || instruction.opcode == "S_CMP_LE_I32" ||
      instruction.opcode == "S_CMP_GT_U32" || instruction.opcode == "S_CMP_GE_U32" ||
      instruction.opcode == "S_CMP_LT_U32" || instruction.opcode == "S_CMP_LE_U32") {
    if (!ValidateOperandCount(instruction, 2, error_message)) {
      return false;
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

    if (instruction.opcode == "S_CMP_EQ_I32") {
      state->scc = BitCast<std::int32_t>(lhs) == BitCast<std::int32_t>(rhs);
    } else if (instruction.opcode == "S_CMP_LG_I32") {
      state->scc = BitCast<std::int32_t>(lhs) != BitCast<std::int32_t>(rhs);
    } else if (instruction.opcode == "S_CMP_GT_I32") {
      state->scc = BitCast<std::int32_t>(lhs) > BitCast<std::int32_t>(rhs);
    } else if (instruction.opcode == "S_CMP_EQ_U32") {
      state->scc = lhs == rhs;
    } else if (instruction.opcode == "S_CMP_LG_U32") {
      state->scc = lhs != rhs;
    } else if (instruction.opcode == "S_CMP_GE_I32") {
      state->scc = BitCast<std::int32_t>(lhs) >= BitCast<std::int32_t>(rhs);
    } else if (instruction.opcode == "S_CMP_LT_I32") {
      state->scc = BitCast<std::int32_t>(lhs) < BitCast<std::int32_t>(rhs);
    } else if (instruction.opcode == "S_CMP_LE_I32") {
      state->scc = BitCast<std::int32_t>(lhs) <= BitCast<std::int32_t>(rhs);
    } else if (instruction.opcode == "S_CMP_GT_U32") {
      state->scc = lhs > rhs;
    } else if (instruction.opcode == "S_CMP_GE_U32") {
      state->scc = lhs >= rhs;
    } else if (instruction.opcode == "S_CMP_LE_U32") {
      state->scc = lhs <= rhs;
    } else {
      state->scc = lhs < rhs;
    }
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }

  if (instruction.opcode == "S_MOV_B32" || instruction.opcode == "S_MOVK_I32") {
    if (!ValidateOperandCount(instruction, 2, error_message)) {
      return false;
    }
    const std::uint32_t value =
        ReadScalarOperand(instruction.operands[1], *state, error_message);
    if (error_message != nullptr && !error_message->empty()) {
      return false;
    }
    return WriteScalarOperand(instruction.operands[0], value, state, error_message);
  }

  if (instruction.opcode == "V_MOV_B32" || instruction.opcode == "V_NOT_B32" ||
      instruction.opcode == "V_BFREV_B32" ||
      instruction.opcode == "V_CLS_I32" ||
      instruction.opcode == "V_CLZ_I32_U32" ||
      instruction.opcode == "V_CTZ_I32_B32" ||
      instruction.opcode == "V_CVT_F32_UBYTE0" ||
      instruction.opcode == "V_CVT_F32_UBYTE1" ||
      instruction.opcode == "V_CVT_F32_UBYTE2" ||
      instruction.opcode == "V_CVT_F32_UBYTE3" ||
      instruction.opcode == "V_CVT_F32_I32" ||
      instruction.opcode == "V_CVT_F32_U32" ||
      instruction.opcode == "V_EXP_F32" ||
      instruction.opcode == "V_LOG_F32" ||
      instruction.opcode == "V_RCP_F32" ||
      instruction.opcode == "V_RCP_IFLAG_F32" ||
      instruction.opcode == "V_RSQ_F32" ||
      instruction.opcode == "V_SQRT_F32" ||
      instruction.opcode == "V_SIN_F32" ||
      instruction.opcode == "V_COS_F32" ||
      instruction.opcode == "V_FREXP_EXP_I32_F32" ||
      instruction.opcode == "V_FREXP_MANT_F32" ||
      instruction.opcode == "V_FRACT_F32" ||
      instruction.opcode == "V_TRUNC_F32" ||
      instruction.opcode == "V_CEIL_F32" ||
      instruction.opcode == "V_RNDNE_F32" ||
      instruction.opcode == "V_FLOOR_F32" ||
      instruction.opcode == "V_CVT_U32_F32" ||
      instruction.opcode == "V_CVT_FLOOR_I32_F32" ||
      instruction.opcode == "V_CVT_NEAREST_I32_F32" ||
      instruction.opcode == "V_CVT_I32_F32") {
    if (!ValidateOperandCount(instruction, 2, error_message)) {
      return false;
    }
    for (std::size_t lane_index = 0; lane_index < state->ActiveLaneCount();
         ++lane_index) {
      if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
        continue;
      }
      const std::uint32_t value = ReadVectorOperand(
          instruction.operands[1], *state, lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      const std::uint32_t result =
          instruction.opcode == "V_MOV_B32"
              ? value
              : EvaluateVectorUnarySeedInstruction(instruction.opcode, value);
      if (!WriteVectorOperand(instruction.operands[0], lane_index, result, state,
                              error_message)) {
        return false;
      }
    }
    return true;
  }

  if (instruction.opcode == "V_CVT_F64_F32" ||
      instruction.opcode == "V_CVT_F64_I32" ||
      instruction.opcode == "V_CVT_F64_U32") {
    if (!ValidateOperandCount(instruction, 2, error_message)) {
      return false;
    }
    for (std::size_t lane_index = 0; lane_index < state->ActiveLaneCount();
         ++lane_index) {
      if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
        continue;
      }
      const std::uint32_t value = ReadVectorOperand(
          instruction.operands[1], *state, lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      const std::uint64_t result =
          EvaluateWideVectorUnarySeedInstruction(instruction.opcode, value);
      if (!WriteWideVectorOperand(instruction.operands[0], lane_index, result,
                                  state, error_message)) {
        return false;
      }
    }
    return true;
  }

  if (instruction.opcode == "V_CVT_F32_F64" ||
      instruction.opcode == "V_CVT_I32_F64" ||
      instruction.opcode == "V_CVT_U32_F64" ||
      instruction.opcode == "V_FREXP_EXP_I32_F64") {
    if (!ValidateOperandCount(instruction, 2, error_message)) {
      return false;
    }
    for (std::size_t lane_index = 0; lane_index < state->ActiveLaneCount();
         ++lane_index) {
      if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
        continue;
      }
      const std::uint64_t value = ReadWideSourceOperand(
          instruction.operands[1], *state, lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      const std::uint32_t result =
          EvaluateVectorUnaryFromWideSeedInstruction(instruction.opcode, value);
      if (!WriteVectorOperand(instruction.operands[0], lane_index, result, state,
                              error_message)) {
        return false;
      }
    }
    return true;
  }

  if (instruction.opcode == "V_RCP_F64" ||
      instruction.opcode == "V_RSQ_F64" ||
      instruction.opcode == "V_SQRT_F64" ||
      instruction.opcode == "V_FREXP_MANT_F64" ||
      instruction.opcode == "V_FRACT_F64" ||
      instruction.opcode == "V_TRUNC_F64" ||
      instruction.opcode == "V_CEIL_F64" ||
      instruction.opcode == "V_RNDNE_F64" ||
      instruction.opcode == "V_FLOOR_F64") {
    if (!ValidateOperandCount(instruction, 2, error_message)) {
      return false;
    }
    for (std::size_t lane_index = 0; lane_index < state->ActiveLaneCount();
         ++lane_index) {
      if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
        continue;
      }
      const std::uint64_t value = ReadWideSourceOperand(
          instruction.operands[1], *state, lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      const std::uint64_t result =
          EvaluateWideVectorUnaryToWideSeedInstruction(instruction.opcode,
                                                       value);
      if (!WriteWideVectorOperand(instruction.operands[0], lane_index, result,
                                  state, error_message)) {
        return false;
      }
    }
    return true;
  }

  if (IsVectorCompareSeedInstruction(instruction.opcode)) {
    if (!ValidateOperandCount(instruction, 3, error_message)) {
      return false;
    }
    if (instruction.operands[0].kind != OperandKind::kSgpr ||
        instruction.operands[0].index != kImplicitVccPairSgprIndex) {
      if (error_message != nullptr) {
        *error_message = "expected implicit VCC destination operand";
      }
      return false;
    }

    const bool writes_exec = IsVectorCmpxSeedInstruction(instruction.opcode);
    const bool is_wide_class =
        IsWideVectorCompareClassSeedInstruction(instruction.opcode);
    const bool is_wide_compare =
        !is_wide_class && IsWideVectorCompareSeedInstruction(instruction.opcode);
    std::uint64_t next_vcc_mask = writes_exec ? 0ULL : state->vcc_mask;
    for (std::size_t lane_index = 0; lane_index < state->ActiveLaneCount();
         ++lane_index) {
      if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
        continue;
      }
      const std::uint64_t lane_bit = 1ULL << lane_index;
      bool lane_result = false;
      if (is_wide_class) {
        const std::uint64_t lhs = ReadWideSourceOperand(
            instruction.operands[1], *state, lane_index, error_message);
        if (error_message != nullptr && !error_message->empty()) {
          return false;
        }
        const std::uint32_t rhs = ReadVectorOperand(instruction.operands[2], *state,
                                                    lane_index, error_message);
        if (error_message != nullptr && !error_message->empty()) {
          return false;
        }
        lane_result = EvaluateWideVectorCompareClassSeedInstruction(
            instruction.opcode, lhs, rhs);
      } else if (is_wide_compare) {
        const std::uint64_t lhs = ReadWideSourceOperand(
            instruction.operands[1], *state, lane_index, error_message);
        if (error_message != nullptr && !error_message->empty()) {
          return false;
        }
        const std::uint64_t rhs = ReadWideSourceOperand(
            instruction.operands[2], *state, lane_index, error_message);
        if (error_message != nullptr && !error_message->empty()) {
          return false;
        }
        lane_result =
            EvaluateWideVectorCompareSeedInstruction(instruction.opcode, lhs, rhs);
      } else {
        const std::uint32_t lhs = ReadVectorOperand(instruction.operands[1], *state,
                                                    lane_index, error_message);
        if (error_message != nullptr && !error_message->empty()) {
          return false;
        }
        const std::uint32_t rhs = ReadVectorOperand(instruction.operands[2], *state,
                                                    lane_index, error_message);
        if (error_message != nullptr && !error_message->empty()) {
          return false;
        }
        lane_result =
            EvaluateVectorCompareSeedInstruction(instruction.opcode, lhs, rhs);
      }
      if (lane_result) {
        next_vcc_mask |= lane_bit;
      } else {
        next_vcc_mask &= ~lane_bit;
      }
    }
    state->vcc_mask = next_vcc_mask;
    if (writes_exec) {
      state->exec_mask = next_vcc_mask;
    }
    state->ClampMasksToLaneCount();
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }

  if (instruction.opcode == "V_ADD_U32" || instruction.opcode == "V_SUB_U32" ||
      instruction.opcode == "V_SUBREV_U32" || instruction.opcode == "V_MIN_I32" ||
      instruction.opcode == "V_MAX_I32" || instruction.opcode == "V_MIN_U32" ||
      instruction.opcode == "V_MAX_U32" ||
      instruction.opcode == "V_CNDMASK_B32" ||
      instruction.opcode == "V_LSHRREV_B32" ||
      instruction.opcode == "V_ASHRREV_I32" ||
      instruction.opcode == "V_LSHLREV_B32" || instruction.opcode == "V_AND_B32" ||
      instruction.opcode == "V_OR_B32" || instruction.opcode == "V_XOR_B32") {
    if (!ValidateOperandCount(instruction, 3, error_message)) {
      return false;
    }
    for (std::size_t lane_index = 0; lane_index < state->ActiveLaneCount();
         ++lane_index) {
      if (((state->exec_mask >> lane_index) & 1ULL) == 0) {
        continue;
      }
      const std::uint32_t lhs = ReadVectorOperand(instruction.operands[1], *state,
                                                  lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      const std::uint32_t rhs = ReadVectorOperand(instruction.operands[2], *state,
                                                  lane_index, error_message);
      if (error_message != nullptr && !error_message->empty()) {
        return false;
      }
      const std::uint32_t result =
          instruction.opcode == "V_CNDMASK_B32"
              ? (((state->vcc_mask >> lane_index) & 1ULL) != 0 ? rhs : lhs)
              : EvaluateVectorBinarySeedInstruction(instruction.opcode, lhs, rhs);
      if (!WriteVectorOperand(instruction.operands[0], lane_index, result, state,
                              error_message)) {
        return false;
      }
    }
    return true;
  }

  if (error_message != nullptr) {
    *error_message = BuildInterpreterBringupMessage(instruction.opcode);
  }
  return false;
}

bool ExecuteCompiledSeedInstruction(const Gfx1201CompiledInstruction& instruction,
                                    WaveExecutionState* state,
                                    bool* pc_was_updated,
                                    std::string* error_message) {
  switch (instruction.opcode) {
    case Gfx1201CompiledOpcode::kSEndpgm:
    case Gfx1201CompiledOpcode::kSNop:
    case Gfx1201CompiledOpcode::kSAddU32:
    case Gfx1201CompiledOpcode::kSAddI32:
    case Gfx1201CompiledOpcode::kSSubU32:
    case Gfx1201CompiledOpcode::kSCmpEqI32:
    case Gfx1201CompiledOpcode::kSCmpLgI32:
    case Gfx1201CompiledOpcode::kSCmpGtI32:
    case Gfx1201CompiledOpcode::kSCmpEqU32:
    case Gfx1201CompiledOpcode::kSCmpLgU32:
    case Gfx1201CompiledOpcode::kSCmpGeI32:
    case Gfx1201CompiledOpcode::kSCmpLtI32:
    case Gfx1201CompiledOpcode::kSCmpLeI32:
    case Gfx1201CompiledOpcode::kSCmpGtU32:
    case Gfx1201CompiledOpcode::kSCmpGeU32:
    case Gfx1201CompiledOpcode::kSCmpLtU32:
    case Gfx1201CompiledOpcode::kSCmpLeU32:
    case Gfx1201CompiledOpcode::kSBranch:
    case Gfx1201CompiledOpcode::kSCbranchScc0:
    case Gfx1201CompiledOpcode::kSCbranchScc1:
    case Gfx1201CompiledOpcode::kSCbranchVccz:
    case Gfx1201CompiledOpcode::kSCbranchVccnz:
    case Gfx1201CompiledOpcode::kSCbranchExecz:
    case Gfx1201CompiledOpcode::kSCbranchExecnz:
    case Gfx1201CompiledOpcode::kSMovB32:
    case Gfx1201CompiledOpcode::kSMovkI32:
    case Gfx1201CompiledOpcode::kVMovB32:
    case Gfx1201CompiledOpcode::kVCmpEqI32:
    case Gfx1201CompiledOpcode::kVCmpNeI32:
    case Gfx1201CompiledOpcode::kVCmpLtI32:
    case Gfx1201CompiledOpcode::kVCmpLeI32:
    case Gfx1201CompiledOpcode::kVCmpGtI32:
    case Gfx1201CompiledOpcode::kVCmpGeI32:
    case Gfx1201CompiledOpcode::kVCmpEqU32:
    case Gfx1201CompiledOpcode::kVCmpNeU32:
    case Gfx1201CompiledOpcode::kVCmpLtU32:
    case Gfx1201CompiledOpcode::kVCmpLeU32:
    case Gfx1201CompiledOpcode::kVCmpGtU32:
    case Gfx1201CompiledOpcode::kVCmpGeU32:
    case Gfx1201CompiledOpcode::kVCmpxEqI32:
    case Gfx1201CompiledOpcode::kVCmpxNeI32:
    case Gfx1201CompiledOpcode::kVCmpxLtI32:
    case Gfx1201CompiledOpcode::kVCmpxLeI32:
    case Gfx1201CompiledOpcode::kVCmpxGtI32:
    case Gfx1201CompiledOpcode::kVCmpxGeI32:
    case Gfx1201CompiledOpcode::kVCmpxEqU32:
    case Gfx1201CompiledOpcode::kVCmpxNeU32:
    case Gfx1201CompiledOpcode::kVCmpxLtU32:
    case Gfx1201CompiledOpcode::kVCmpxLeU32:
    case Gfx1201CompiledOpcode::kVCmpxGtU32:
    case Gfx1201CompiledOpcode::kVCmpxGeU32:
    case Gfx1201CompiledOpcode::kVCmpEqI64:
    case Gfx1201CompiledOpcode::kVCmpNeI64:
    case Gfx1201CompiledOpcode::kVCmpLtI64:
    case Gfx1201CompiledOpcode::kVCmpLeI64:
    case Gfx1201CompiledOpcode::kVCmpGtI64:
    case Gfx1201CompiledOpcode::kVCmpGeI64:
    case Gfx1201CompiledOpcode::kVCmpEqU64:
    case Gfx1201CompiledOpcode::kVCmpNeU64:
    case Gfx1201CompiledOpcode::kVCmpLtU64:
    case Gfx1201CompiledOpcode::kVCmpLeU64:
    case Gfx1201CompiledOpcode::kVCmpGtU64:
    case Gfx1201CompiledOpcode::kVCmpGeU64:
    case Gfx1201CompiledOpcode::kVCmpxEqI64:
    case Gfx1201CompiledOpcode::kVCmpxNeI64:
    case Gfx1201CompiledOpcode::kVCmpxLtI64:
    case Gfx1201CompiledOpcode::kVCmpxLeI64:
    case Gfx1201CompiledOpcode::kVCmpxGtI64:
    case Gfx1201CompiledOpcode::kVCmpxGeI64:
    case Gfx1201CompiledOpcode::kVCmpxEqU64:
    case Gfx1201CompiledOpcode::kVCmpxNeU64:
    case Gfx1201CompiledOpcode::kVCmpxLtU64:
    case Gfx1201CompiledOpcode::kVCmpxLeU64:
    case Gfx1201CompiledOpcode::kVCmpxGtU64:
    case Gfx1201CompiledOpcode::kVCmpxGeU64:
    case Gfx1201CompiledOpcode::kVCmpClassF32:
    case Gfx1201CompiledOpcode::kVCmpEqF32:
    case Gfx1201CompiledOpcode::kVCmpGeF32:
    case Gfx1201CompiledOpcode::kVCmpGtF32:
    case Gfx1201CompiledOpcode::kVCmpLeF32:
    case Gfx1201CompiledOpcode::kVCmpLgF32:
    case Gfx1201CompiledOpcode::kVCmpLtF32:
    case Gfx1201CompiledOpcode::kVCmpNeqF32:
    case Gfx1201CompiledOpcode::kVCmpOF32:
    case Gfx1201CompiledOpcode::kVCmpUF32:
    case Gfx1201CompiledOpcode::kVCmpNgeF32:
    case Gfx1201CompiledOpcode::kVCmpNlgF32:
    case Gfx1201CompiledOpcode::kVCmpNgtF32:
    case Gfx1201CompiledOpcode::kVCmpNleF32:
    case Gfx1201CompiledOpcode::kVCmpNltF32:
    case Gfx1201CompiledOpcode::kVCmpxClassF32:
    case Gfx1201CompiledOpcode::kVCmpxEqF32:
    case Gfx1201CompiledOpcode::kVCmpxGeF32:
    case Gfx1201CompiledOpcode::kVCmpxGtF32:
    case Gfx1201CompiledOpcode::kVCmpxLeF32:
    case Gfx1201CompiledOpcode::kVCmpxLgF32:
    case Gfx1201CompiledOpcode::kVCmpxLtF32:
    case Gfx1201CompiledOpcode::kVCmpxNeqF32:
    case Gfx1201CompiledOpcode::kVCmpxOF32:
    case Gfx1201CompiledOpcode::kVCmpxUF32:
    case Gfx1201CompiledOpcode::kVCmpxNgeF32:
    case Gfx1201CompiledOpcode::kVCmpxNlgF32:
    case Gfx1201CompiledOpcode::kVCmpxNgtF32:
    case Gfx1201CompiledOpcode::kVCmpxNleF32:
    case Gfx1201CompiledOpcode::kVCmpxNltF32:
    case Gfx1201CompiledOpcode::kVCmpClassF64:
    case Gfx1201CompiledOpcode::kVCmpEqF64:
    case Gfx1201CompiledOpcode::kVCmpGeF64:
    case Gfx1201CompiledOpcode::kVCmpGtF64:
    case Gfx1201CompiledOpcode::kVCmpLeF64:
    case Gfx1201CompiledOpcode::kVCmpLgF64:
    case Gfx1201CompiledOpcode::kVCmpLtF64:
    case Gfx1201CompiledOpcode::kVCmpNeqF64:
    case Gfx1201CompiledOpcode::kVCmpOF64:
    case Gfx1201CompiledOpcode::kVCmpUF64:
    case Gfx1201CompiledOpcode::kVCmpNgeF64:
    case Gfx1201CompiledOpcode::kVCmpNlgF64:
    case Gfx1201CompiledOpcode::kVCmpNgtF64:
    case Gfx1201CompiledOpcode::kVCmpNleF64:
    case Gfx1201CompiledOpcode::kVCmpNltF64:
    case Gfx1201CompiledOpcode::kVCmpxClassF64:
    case Gfx1201CompiledOpcode::kVCmpxEqF64:
    case Gfx1201CompiledOpcode::kVCmpxGeF64:
    case Gfx1201CompiledOpcode::kVCmpxGtF64:
    case Gfx1201CompiledOpcode::kVCmpxLeF64:
    case Gfx1201CompiledOpcode::kVCmpxLgF64:
    case Gfx1201CompiledOpcode::kVCmpxLtF64:
    case Gfx1201CompiledOpcode::kVCmpxNeqF64:
    case Gfx1201CompiledOpcode::kVCmpxOF64:
    case Gfx1201CompiledOpcode::kVCmpxUF64:
    case Gfx1201CompiledOpcode::kVCmpxNgeF64:
    case Gfx1201CompiledOpcode::kVCmpxNlgF64:
    case Gfx1201CompiledOpcode::kVCmpxNgtF64:
    case Gfx1201CompiledOpcode::kVCmpxNleF64:
    case Gfx1201CompiledOpcode::kVCmpxNltF64:
    case Gfx1201CompiledOpcode::kVNotB32:
    case Gfx1201CompiledOpcode::kVBfrevB32:
    case Gfx1201CompiledOpcode::kVClsI32:
    case Gfx1201CompiledOpcode::kVClzI32U32:
    case Gfx1201CompiledOpcode::kVCtzI32B32:
    case Gfx1201CompiledOpcode::kVCvtF32Ubyte0:
    case Gfx1201CompiledOpcode::kVCvtF32Ubyte1:
    case Gfx1201CompiledOpcode::kVCvtF32Ubyte2:
    case Gfx1201CompiledOpcode::kVCvtF32Ubyte3:
    case Gfx1201CompiledOpcode::kVCvtF32I32:
    case Gfx1201CompiledOpcode::kVCvtF32U32:
    case Gfx1201CompiledOpcode::kVCvtF32F64:
    case Gfx1201CompiledOpcode::kVCvtF64F32:
    case Gfx1201CompiledOpcode::kVCvtF64I32:
    case Gfx1201CompiledOpcode::kVCvtF64U32:
    case Gfx1201CompiledOpcode::kVCvtU32F32:
    case Gfx1201CompiledOpcode::kVCvtU32F64:
    case Gfx1201CompiledOpcode::kVCvtI32F32:
    case Gfx1201CompiledOpcode::kVCvtFloorI32F32:
    case Gfx1201CompiledOpcode::kVCvtNearestI32F32:
    case Gfx1201CompiledOpcode::kVCvtI32F64:
    case Gfx1201CompiledOpcode::kVExpF32:
    case Gfx1201CompiledOpcode::kVLogF32:
    case Gfx1201CompiledOpcode::kVRcpF32:
    case Gfx1201CompiledOpcode::kVRcpIflagF32:
    case Gfx1201CompiledOpcode::kVRsqF32:
    case Gfx1201CompiledOpcode::kVSqrtF32:
    case Gfx1201CompiledOpcode::kVSinF32:
    case Gfx1201CompiledOpcode::kVCosF32:
    case Gfx1201CompiledOpcode::kVRcpF64:
    case Gfx1201CompiledOpcode::kVRsqF64:
    case Gfx1201CompiledOpcode::kVSqrtF64:
    case Gfx1201CompiledOpcode::kVFrexpExpI32F32:
    case Gfx1201CompiledOpcode::kVFrexpMantF32:
    case Gfx1201CompiledOpcode::kVFractF32:
    case Gfx1201CompiledOpcode::kVFrexpExpI32F64:
    case Gfx1201CompiledOpcode::kVFrexpMantF64:
    case Gfx1201CompiledOpcode::kVFractF64:
    case Gfx1201CompiledOpcode::kVTruncF32:
    case Gfx1201CompiledOpcode::kVCeilF32:
    case Gfx1201CompiledOpcode::kVRndneF32:
    case Gfx1201CompiledOpcode::kVFloorF32:
    case Gfx1201CompiledOpcode::kVTruncF64:
    case Gfx1201CompiledOpcode::kVCeilF64:
    case Gfx1201CompiledOpcode::kVRndneF64:
    case Gfx1201CompiledOpcode::kVFloorF64:
    case Gfx1201CompiledOpcode::kVAddU32:
    case Gfx1201CompiledOpcode::kVSubU32:
    case Gfx1201CompiledOpcode::kVSubrevU32:
    case Gfx1201CompiledOpcode::kVMinI32:
    case Gfx1201CompiledOpcode::kVMaxI32:
    case Gfx1201CompiledOpcode::kVMinU32:
    case Gfx1201CompiledOpcode::kVMaxU32:
    case Gfx1201CompiledOpcode::kVCndmaskB32:
    case Gfx1201CompiledOpcode::kVLshrrevB32:
    case Gfx1201CompiledOpcode::kVAshrrevI32:
    case Gfx1201CompiledOpcode::kVLshlrevB32:
    case Gfx1201CompiledOpcode::kVAndB32:
    case Gfx1201CompiledOpcode::kVOrB32:
    case Gfx1201CompiledOpcode::kVXorB32:
      return ExecuteDecodedSeedInstruction(instruction.decoded_instruction, state,
                                           pc_was_updated, error_message);
    case Gfx1201CompiledOpcode::kUnknown:
      break;
  }

  if (error_message != nullptr) {
    *error_message = "unknown gfx1201 compiled opcode";
  }
  return false;
}

template <typename ProgramT>
bool ValidateStateAndResetForExecution(WaveExecutionState* state,
                                       std::string* error_message) {
  if (state == nullptr) {
    if (error_message != nullptr) {
      *error_message = "wave execution state must not be null";
    }
    return false;
  }
  if (state->pc == 0) {
    state->halted = false;
    state->waiting_on_barrier = false;
  }
  state->SetLaneCount(kGfx1201LaneCount);
  return true;
}

}  // namespace

bool Gfx1201Interpreter::Supports(std::string_view opcode) const {
  const std::string_view normalized_opcode =
      NormalizeExecutableSeedOpcode(opcode);
  return FindGfx1201InstructionSupport(ImportedSupportOpcode(normalized_opcode)) !=
             nullptr &&
         IsExecutableSeedOpcode(normalized_opcode);
}

bool Gfx1201Interpreter::CompileProgram(
    std::span<const DecodedInstruction> program,
    std::vector<Gfx1201CompiledInstruction>* compiled_program,
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
    const std::string_view normalized_opcode =
        NormalizeExecutableSeedOpcode(instruction.opcode);
    if (FindGfx1201InstructionSupport(ImportedSupportOpcode(normalized_opcode)) ==
        nullptr) {
      if (error_message != nullptr) {
        *error_message = "unknown gfx1201 opcode";
      }
      return false;
    }

    Gfx1201CompiledInstruction compiled_instruction;
    compiled_instruction.decoded_instruction = instruction;
    compiled_instruction.decoded_instruction.opcode = normalized_opcode;
    if (!TryCompileExecutableOpcode(normalized_opcode,
                                    &compiled_instruction.opcode)) {
      if (error_message != nullptr) {
        *error_message = BuildInterpreterBringupMessage(normalized_opcode);
      }
      return false;
    }
    compiled_program->push_back(compiled_instruction);
  }

  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool Gfx1201Interpreter::ExecuteProgram(std::span<const DecodedInstruction> program,
                                        WaveExecutionState* state,
                                        std::string* error_message) const {
  return ExecuteProgram(program, state, nullptr, error_message);
}

bool Gfx1201Interpreter::ExecuteProgram(std::span<const DecodedInstruction> program,
                                        WaveExecutionState* state,
                                        ExecutionMemory* memory,
                                        std::string* error_message) const {
  (void)memory;
  std::vector<Gfx1201CompiledInstruction> compiled_program;
  if (!CompileProgram(program, &compiled_program, error_message)) {
    return false;
  }
  return ExecuteProgram(compiled_program, state, memory, error_message);
}

bool Gfx1201Interpreter::ExecuteProgram(
    std::span<const Gfx1201CompiledInstruction> program,
    WaveExecutionState* state,
    std::string* error_message) const {
  return ExecuteProgram(program, state, nullptr, error_message);
}

bool Gfx1201Interpreter::ExecuteProgram(
    std::span<const Gfx1201CompiledInstruction> program,
    WaveExecutionState* state,
    ExecutionMemory* memory,
    std::string* error_message) const {
  (void)memory;
  if (!ValidateStateAndResetForExecution<std::span<const Gfx1201CompiledInstruction>>(
          state, error_message)) {
    return false;
  }

  while (!state->halted && state->pc < program.size()) {
    const std::uint64_t next_pc = state->pc + 1;
    const Gfx1201CompiledInstruction& instruction =
        program[static_cast<std::size_t>(state->pc)];
    bool pc_was_updated = false;
    if (!ExecuteCompiledSeedInstruction(instruction, state, &pc_was_updated,
                                        error_message)) {
      return false;
    }
    if (!state->halted && !pc_was_updated) {
      state->pc = next_pc;
    }
  }

  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

std::span<const std::string_view> Gfx1201Interpreter::ExecutableSeedOpcodes()
    const {
  return kExecutableSeedOpcodes;
}

std::span<const Gfx1201FamilyFocus> Gfx1201Interpreter::CarryOverFamilyFocus()
    const {
  return GetGfx1201CarryOverFamilyFocus();
}

std::span<const Gfx1201FamilyFocus> Gfx1201Interpreter::Rdna4DeltaFamilyFocus()
    const {
  return GetGfx1201Rdna4DeltaFamilyFocus();
}

std::string_view Gfx1201Interpreter::BringupStatus() const {
  return DescribeGfx1201BringupPhase();
}

}  // namespace mirage::sim::isa
