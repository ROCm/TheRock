#include "lib/sim/isa/gfx950/binary_decoder.h"

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "lib/sim/isa/instruction_catalog.h"

namespace mirage::sim::isa {
namespace {

constexpr std::uint32_t kEncSopp = 0x17f;
constexpr std::uint32_t kEncSopc = 0x17e;
constexpr std::uint32_t kEncSop1 = 0x17d;
constexpr std::uint32_t kEncSop2 = 0x2;
constexpr std::uint32_t kEncSopk = 0xb;
constexpr std::uint32_t kEncDs = 0x36;
constexpr std::uint32_t kEncFlat = 55;
constexpr std::uint32_t kEncSmem = 0x30;
constexpr std::uint32_t kEncVop1 = 0x3f;
constexpr std::uint32_t kEncVopc = 0x3e;
constexpr std::uint32_t kEncVop2 = 0x0;
constexpr std::uint32_t kEncVop3 = 0x34;
constexpr std::uint16_t kImplicitVccPairSgprIndex = 106;

constexpr std::uint32_t ExtractBits(std::uint32_t value,
                                    std::uint32_t bit_offset,
                                    std::uint32_t bit_count) {
  if (bit_count == 32) {
    return value;
  }
  return (value >> bit_offset) & ((1u << bit_count) - 1u);
}

constexpr std::uint64_t ExtractBits(std::uint64_t value,
                                    std::uint32_t bit_offset,
                                    std::uint32_t bit_count) {
  if (bit_count == 64) {
    return value;
  }
  return (value >> bit_offset) & ((1ULL << bit_count) - 1ULL);
}

bool IsInlineInteger(std::uint32_t raw_value) {
  return raw_value >= 128 && raw_value <= 208;
}

std::int32_t SignExtend16(std::uint32_t value) {
  return static_cast<std::int32_t>(static_cast<std::int16_t>(value & 0xffffu));
}

std::int32_t SignExtend13(std::uint32_t value) {
  const std::uint32_t masked = value & 0x1fffu;
  if ((masked & (1u << 12)) == 0) {
    return static_cast<std::int32_t>(masked);
  }
  return static_cast<std::int32_t>(masked | 0xffffe000u);
}

std::int32_t SignExtend21(std::uint32_t value) {
  const std::uint32_t masked = value & 0x1fffffu;
  if ((masked & (1u << 20)) == 0) {
    return static_cast<std::int32_t>(masked);
  }
  return static_cast<std::int32_t>(masked | 0xffe00000u);
}

std::uint32_t DecodeInlineInteger(std::uint32_t raw_value) {
  if (raw_value <= 192) {
    return raw_value - 128;
  }
  const std::int32_t signed_value =
      -static_cast<std::int32_t>(raw_value - 192);
  return static_cast<std::uint32_t>(signed_value);
}

bool IsFlatInstructionWord(std::uint32_t word) {
  return ExtractBits(word, 26, 6) == kEncFlat && ExtractBits(word, 14, 2) == 0;
}

bool IsDsInstructionWord(std::uint32_t word) {
  return ExtractBits(word, 26, 6) == kEncDs;
}

bool IsVectorCarryOutBinaryOpcode(std::string_view opcode) {
  return opcode == "V_ADD_CO_U32" || opcode == "V_SUB_CO_U32" ||
         opcode == "V_SUBREV_CO_U32";
}

bool IsVectorCarryInBinaryOpcode(std::string_view opcode) {
  return opcode == "V_ADDC_CO_U32" || opcode == "V_SUBB_CO_U32" ||
         opcode == "V_SUBBREV_CO_U32";
}

bool IsSupportedVectorCompareOpcode(std::string_view opcode) {
  return opcode == "V_CMP_F_F32" || opcode == "V_CMP_LT_F32" ||
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

bool IsSupportedPromotedVop3UnaryOpcode(std::string_view opcode) {
  return opcode == "V_MOV_B64" ||
         opcode == "V_READFIRSTLANE_B32" ||
         opcode == "V_NOT_B32" || opcode == "V_BFREV_B32" ||
         opcode == "V_FFBH_U32" || opcode == "V_FFBL_B32" ||
         opcode == "V_FFBH_I32" ||
         opcode == "V_CVT_F32_I32" || opcode == "V_CVT_F32_U32" ||
         opcode == "V_CVT_U32_F32" || opcode == "V_CVT_I32_F32" ||
         opcode == "V_CVT_I32_F64" || opcode == "V_CVT_U32_F64" ||
         opcode == "V_CVT_F16_F32" || opcode == "V_CVT_F32_F16" ||
         opcode == "V_CVT_F32_F64" || opcode == "V_CVT_F64_F32" ||
         opcode == "V_CVT_F64_I32" || opcode == "V_CVT_F64_U32";
}

bool IsSupportedPromotedVop3BinaryOpcode(std::string_view opcode) {
  return opcode == "V_ADD_U32" || opcode == "V_SUB_U32" ||
         opcode == "V_SUBREV_U32" ||
         opcode == "V_ADD_F32" || opcode == "V_SUB_F32" ||
         opcode == "V_MUL_F32" || opcode == "V_MIN_F32" ||
         opcode == "V_MAX_F32" ||
         opcode == "V_MIN_I32" || opcode == "V_MAX_I32" ||
         opcode == "V_MIN_U32" || opcode == "V_MAX_U32" ||
         opcode == "V_LSHRREV_B32" || opcode == "V_ASHRREV_I32" ||
         opcode == "V_LSHLREV_B32" ||
         opcode == "V_AND_B32" || opcode == "V_OR_B32" ||
         opcode == "V_XOR_B32";
}

bool IsFlatGlobalInstructionWord(std::uint32_t word) {
  return ExtractBits(word, 26, 6) == kEncFlat && ExtractBits(word, 14, 2) == 2;
}

bool IsSupportedFlatVectorMemoryOpcode(std::string_view opcode_name) {
  return opcode_name == "FLAT_LOAD_UBYTE" || opcode_name == "FLAT_LOAD_SBYTE" ||
         opcode_name == "FLAT_LOAD_UBYTE_D16" ||
         opcode_name == "FLAT_LOAD_UBYTE_D16_HI" ||
         opcode_name == "FLAT_LOAD_SBYTE_D16" ||
         opcode_name == "FLAT_LOAD_SBYTE_D16_HI" ||
         opcode_name == "FLAT_LOAD_USHORT" || opcode_name == "FLAT_LOAD_SSHORT" ||
         opcode_name == "FLAT_LOAD_SHORT_D16" ||
         opcode_name == "FLAT_LOAD_SHORT_D16_HI" ||
         opcode_name == "FLAT_LOAD_DWORD" || opcode_name == "FLAT_LOAD_DWORDX2" ||
         opcode_name == "FLAT_LOAD_DWORDX3" || opcode_name == "FLAT_LOAD_DWORDX4" ||
         opcode_name == "FLAT_STORE_BYTE" ||
         opcode_name == "FLAT_STORE_BYTE_D16_HI" ||
         opcode_name == "FLAT_STORE_SHORT" ||
         opcode_name == "FLAT_STORE_SHORT_D16_HI" ||
         opcode_name == "FLAT_STORE_DWORD" || opcode_name == "FLAT_STORE_DWORDX2" ||
         opcode_name == "FLAT_STORE_DWORDX3" || opcode_name == "FLAT_STORE_DWORDX4";
}

bool IsSupportedGlobalVectorMemoryOpcode(std::string_view opcode_name) {
  return opcode_name == "GLOBAL_LOAD_UBYTE" ||
         opcode_name == "GLOBAL_LOAD_UBYTE_D16" ||
         opcode_name == "GLOBAL_LOAD_UBYTE_D16_HI" ||
         opcode_name == "GLOBAL_LOAD_SBYTE" ||
         opcode_name == "GLOBAL_LOAD_SBYTE_D16" ||
         opcode_name == "GLOBAL_LOAD_SBYTE_D16_HI" ||
         opcode_name == "GLOBAL_LOAD_USHORT" ||
         opcode_name == "GLOBAL_LOAD_SHORT_D16" ||
         opcode_name == "GLOBAL_LOAD_SHORT_D16_HI" ||
         opcode_name == "GLOBAL_LOAD_SSHORT" ||
         opcode_name == "GLOBAL_LOAD_DWORD" ||
         opcode_name == "GLOBAL_LOAD_DWORDX2" ||
         opcode_name == "GLOBAL_LOAD_DWORDX3" ||
         opcode_name == "GLOBAL_LOAD_DWORDX4" ||
         opcode_name == "GLOBAL_STORE_BYTE" ||
         opcode_name == "GLOBAL_STORE_BYTE_D16_HI" ||
         opcode_name == "GLOBAL_STORE_SHORT" ||
         opcode_name == "GLOBAL_STORE_SHORT_D16_HI" ||
         opcode_name == "GLOBAL_STORE_DWORD" ||
         opcode_name == "GLOBAL_STORE_DWORDX2" ||
         opcode_name == "GLOBAL_STORE_DWORDX3" ||
         opcode_name == "GLOBAL_STORE_DWORDX4";
}

bool IsSupportedDsOpcode(std::string_view opcode_name) {
  return opcode_name == "DS_NOP" ||
         opcode_name == "DS_WRITE_B32" || opcode_name == "DS_READ_B32" ||
         opcode_name == "DS_SWIZZLE_B32" ||
         opcode_name == "DS_PERMUTE_B32" ||
         opcode_name == "DS_BPERMUTE_B32" ||
         opcode_name == "DS_ADD_U32" || opcode_name == "DS_SUB_U32" ||
         opcode_name == "DS_RSUB_U32" || opcode_name == "DS_INC_U32" ||
         opcode_name == "DS_DEC_U32" || opcode_name == "DS_MIN_I32" ||
         opcode_name == "DS_MAX_I32" || opcode_name == "DS_MIN_U32" ||
         opcode_name == "DS_MAX_U32" || opcode_name == "DS_AND_B32" ||
         opcode_name == "DS_OR_B32" || opcode_name == "DS_XOR_B32" ||
         opcode_name == "DS_MSKOR_B32" ||
         opcode_name == "DS_CMPST_B32" ||
         opcode_name == "DS_CMPST_F32" ||
         opcode_name == "DS_ADD_F32" || opcode_name == "DS_MIN_F32" ||
         opcode_name == "DS_MAX_F32" ||
         opcode_name == "DS_PK_ADD_F16" ||
         opcode_name == "DS_PK_ADD_BF16" ||
         opcode_name == "DS_WRITE_ADDTID_B32" ||
         opcode_name == "DS_CONSUME" ||
         opcode_name == "DS_APPEND" ||
         opcode_name == "DS_WRITE_B8" ||
         opcode_name == "DS_WRITE_B16" ||
         opcode_name == "DS_WRITE_B8_D16_HI" ||
         opcode_name == "DS_WRITE_B16_D16_HI" ||
         opcode_name == "DS_WRITE_B96" ||
         opcode_name == "DS_WRITE_B128" ||
         opcode_name == "DS_ADD_U64" ||
         opcode_name == "DS_SUB_U64" ||
         opcode_name == "DS_RSUB_U64" ||
         opcode_name == "DS_INC_U64" ||
         opcode_name == "DS_DEC_U64" ||
         opcode_name == "DS_MIN_I64" ||
         opcode_name == "DS_MAX_I64" ||
         opcode_name == "DS_MIN_U64" ||
         opcode_name == "DS_MAX_U64" ||
         opcode_name == "DS_AND_B64" ||
         opcode_name == "DS_OR_B64" ||
         opcode_name == "DS_XOR_B64" ||
         opcode_name == "DS_MSKOR_B64" ||
         opcode_name == "DS_WRITE_B64" ||
         opcode_name == "DS_WRITE2_B32" ||
         opcode_name == "DS_WRITE2ST64_B32" ||
         opcode_name == "DS_WRITE2_B64" ||
         opcode_name == "DS_WRITE2ST64_B64" ||
         opcode_name == "DS_CMPST_B64" ||
         opcode_name == "DS_CMPST_F64" ||
         opcode_name == "DS_ADD_F64" ||
         opcode_name == "DS_MIN_F64" ||
         opcode_name == "DS_MAX_F64" ||
         opcode_name == "DS_READ2_B32" ||
         opcode_name == "DS_READ2ST64_B32" ||
         opcode_name == "DS_READ_B64" ||
         opcode_name == "DS_READ_B96" ||
         opcode_name == "DS_READ_B128" ||
         opcode_name == "DS_READ_ADDTID_B32" ||
         opcode_name == "DS_READ2_B64" ||
         opcode_name == "DS_READ2ST64_B64" ||
         opcode_name == "DS_READ_I8" ||
         opcode_name == "DS_READ_U8" ||
         opcode_name == "DS_READ_I16" ||
         opcode_name == "DS_READ_U16" ||
         opcode_name == "DS_READ_U8_D16" ||
         opcode_name == "DS_READ_U8_D16_HI" ||
         opcode_name == "DS_READ_I8_D16" ||
         opcode_name == "DS_READ_I8_D16_HI" ||
         opcode_name == "DS_READ_U16_D16" ||
         opcode_name == "DS_READ_U16_D16_HI" ||
         opcode_name == "DS_ADD_RTN_U32" ||
         opcode_name == "DS_SUB_RTN_U32" ||
         opcode_name == "DS_RSUB_RTN_U32" ||
         opcode_name == "DS_INC_RTN_U32" ||
         opcode_name == "DS_DEC_RTN_U32" ||
         opcode_name == "DS_MIN_RTN_I32" ||
         opcode_name == "DS_MAX_RTN_I32" ||
         opcode_name == "DS_MIN_RTN_U32" ||
         opcode_name == "DS_MAX_RTN_U32" ||
         opcode_name == "DS_AND_RTN_B32" ||
         opcode_name == "DS_OR_RTN_B32" ||
         opcode_name == "DS_XOR_RTN_B32" ||
         opcode_name == "DS_MSKOR_RTN_B32" ||
         opcode_name == "DS_WRXCHG_RTN_B32" ||
         opcode_name == "DS_WRXCHG2_RTN_B32" ||
         opcode_name == "DS_WRXCHG2ST64_RTN_B32" ||
         opcode_name == "DS_CMPST_RTN_B32" ||
         opcode_name == "DS_CMPST_RTN_F32" ||
         opcode_name == "DS_WRAP_RTN_B32" ||
         opcode_name == "DS_ADD_RTN_F32" ||
         opcode_name == "DS_MIN_RTN_F32" ||
         opcode_name == "DS_MAX_RTN_F32" ||
         opcode_name == "DS_PK_ADD_RTN_F16" ||
         opcode_name == "DS_PK_ADD_RTN_BF16" ||
         opcode_name == "DS_ADD_RTN_U64" ||
         opcode_name == "DS_SUB_RTN_U64" ||
         opcode_name == "DS_RSUB_RTN_U64" ||
         opcode_name == "DS_INC_RTN_U64" ||
         opcode_name == "DS_DEC_RTN_U64" ||
         opcode_name == "DS_MIN_RTN_I64" ||
         opcode_name == "DS_MAX_RTN_I64" ||
         opcode_name == "DS_MIN_RTN_U64" ||
         opcode_name == "DS_MAX_RTN_U64" ||
         opcode_name == "DS_AND_RTN_B64" ||
         opcode_name == "DS_OR_RTN_B64" ||
         opcode_name == "DS_XOR_RTN_B64" ||
         opcode_name == "DS_MSKOR_RTN_B64" ||
         opcode_name == "DS_WRXCHG_RTN_B64" ||
         opcode_name == "DS_WRXCHG2_RTN_B64" ||
         opcode_name == "DS_WRXCHG2ST64_RTN_B64" ||
         opcode_name == "DS_CMPST_RTN_B64" ||
         opcode_name == "DS_CMPST_RTN_F64" ||
         opcode_name == "DS_CONDXCHG32_RTN_B64" ||
         opcode_name == "DS_ADD_RTN_F64" ||
         opcode_name == "DS_MIN_RTN_F64" ||
         opcode_name == "DS_MAX_RTN_F64";
}

bool IsDsPairWriteOpcode(std::string_view opcode_name) {
  return opcode_name == "DS_WRITE2_B32" ||
         opcode_name == "DS_WRITE2ST64_B32" ||
         opcode_name == "DS_WRITE2_B64" ||
         opcode_name == "DS_WRITE2ST64_B64";
}

bool IsDsPairReadOpcode(std::string_view opcode_name) {
  return opcode_name == "DS_READ2_B32" ||
         opcode_name == "DS_READ2ST64_B32" ||
         opcode_name == "DS_READ2_B64" ||
         opcode_name == "DS_READ2ST64_B64";
}

bool IsDsPairReturnOpcode(std::string_view opcode_name) {
  return opcode_name == "DS_WRXCHG2_RTN_B32" ||
         opcode_name == "DS_WRXCHG2ST64_RTN_B32" ||
         opcode_name == "DS_WRXCHG2_RTN_B64" ||
         opcode_name == "DS_WRXCHG2ST64_RTN_B64";
}

bool IsDsNarrowReadOpcode(std::string_view opcode_name) {
  return opcode_name == "DS_READ_I8" || opcode_name == "DS_READ_U8" ||
         opcode_name == "DS_READ_I16" || opcode_name == "DS_READ_U16" ||
         opcode_name == "DS_READ_U8_D16" ||
         opcode_name == "DS_READ_U8_D16_HI" ||
         opcode_name == "DS_READ_I8_D16" ||
         opcode_name == "DS_READ_I8_D16_HI" ||
         opcode_name == "DS_READ_U16_D16" ||
         opcode_name == "DS_READ_U16_D16_HI";
}

bool IsDsSwizzleOpcode(std::string_view opcode_name) {
  return opcode_name == "DS_SWIZZLE_B32";
}

bool IsDsPermuteOpcode(std::string_view opcode_name) {
  return opcode_name == "DS_PERMUTE_B32" ||
         opcode_name == "DS_BPERMUTE_B32";
}

bool IsDsLaneRoutingOpcode(std::string_view opcode_name) {
  return IsDsSwizzleOpcode(opcode_name) || IsDsPermuteOpcode(opcode_name);
}

bool IsDsDirectReadOpcode(std::string_view opcode_name) {
  return opcode_name == "DS_READ_B32" || opcode_name == "DS_READ_B64" ||
         opcode_name == "DS_READ_B96" || opcode_name == "DS_READ_B128" ||
         IsDsNarrowReadOpcode(opcode_name);
}

bool IsDsAddTidWriteOpcode(std::string_view opcode_name) {
  return opcode_name == "DS_WRITE_ADDTID_B32";
}

bool IsDsAddTidReadOpcode(std::string_view opcode_name) {
  return opcode_name == "DS_READ_ADDTID_B32";
}

bool IsDsWaveCounterOpcode(std::string_view opcode_name) {
  return opcode_name == "DS_CONSUME" || opcode_name == "DS_APPEND";
}

bool IsDsDualDataOpcode(std::string_view opcode_name) {
  return opcode_name == "DS_MSKOR_B32" ||
         opcode_name == "DS_CMPST_B32" ||
         opcode_name == "DS_CMPST_F32" ||
         opcode_name == "DS_MSKOR_B64" ||
         opcode_name == "DS_CMPST_B64" ||
         opcode_name == "DS_CMPST_F64";
}

bool IsDsReturnOpcode(std::string_view opcode_name) {
  return opcode_name == "DS_ADD_RTN_U32" ||
         opcode_name == "DS_SUB_RTN_U32" ||
         opcode_name == "DS_RSUB_RTN_U32" ||
         opcode_name == "DS_INC_RTN_U32" ||
         opcode_name == "DS_DEC_RTN_U32" ||
         opcode_name == "DS_MIN_RTN_I32" ||
         opcode_name == "DS_MAX_RTN_I32" ||
         opcode_name == "DS_MIN_RTN_U32" ||
         opcode_name == "DS_MAX_RTN_U32" ||
         opcode_name == "DS_AND_RTN_B32" ||
         opcode_name == "DS_OR_RTN_B32" ||
         opcode_name == "DS_XOR_RTN_B32" ||
         opcode_name == "DS_WRXCHG_RTN_B32" ||
         opcode_name == "DS_ADD_RTN_F32" ||
         opcode_name == "DS_MIN_RTN_F32" ||
         opcode_name == "DS_MAX_RTN_F32" ||
         opcode_name == "DS_PK_ADD_RTN_F16" ||
         opcode_name == "DS_PK_ADD_RTN_BF16" ||
         opcode_name == "DS_ADD_RTN_U64" ||
         opcode_name == "DS_SUB_RTN_U64" ||
         opcode_name == "DS_RSUB_RTN_U64" ||
         opcode_name == "DS_INC_RTN_U64" ||
         opcode_name == "DS_DEC_RTN_U64" ||
         opcode_name == "DS_MIN_RTN_I64" ||
         opcode_name == "DS_MAX_RTN_I64" ||
         opcode_name == "DS_MIN_RTN_U64" ||
         opcode_name == "DS_MAX_RTN_U64" ||
         opcode_name == "DS_AND_RTN_B64" ||
         opcode_name == "DS_OR_RTN_B64" ||
         opcode_name == "DS_XOR_RTN_B64" ||
         opcode_name == "DS_WRXCHG_RTN_B64" ||
         opcode_name == "DS_CONDXCHG32_RTN_B64" ||
         opcode_name == "DS_ADD_RTN_F64" ||
         opcode_name == "DS_MIN_RTN_F64" ||
         opcode_name == "DS_MAX_RTN_F64";
}

bool IsDsDualDataReturnOpcode(std::string_view opcode_name) {
  return opcode_name == "DS_MSKOR_RTN_B32" ||
         opcode_name == "DS_CMPST_RTN_B32" ||
         opcode_name == "DS_CMPST_RTN_F32" ||
         opcode_name == "DS_WRAP_RTN_B32" ||
         opcode_name == "DS_MSKOR_RTN_B64" ||
         opcode_name == "DS_CMPST_RTN_B64" ||
         opcode_name == "DS_CMPST_RTN_F64";
}

}  // namespace

bool Gfx950BinaryDecoder::DecodeInstruction(
    std::span<const std::uint32_t> words,
    DecodedInstruction* instruction,
    std::size_t* words_consumed,
    std::string* error_message) const {
  if (instruction == nullptr || words_consumed == nullptr) {
    if (error_message != nullptr) {
      *error_message = "decode outputs must not be null";
    }
    return false;
  }
  if (words.empty()) {
    if (error_message != nullptr) {
      *error_message = "instruction stream is empty";
    }
    return false;
  }

  const std::uint32_t word = words.front();
  if (words.size() >= 2 && IsFlatGlobalInstructionWord(word)) {
    return DecodeFlatGlobal(words, instruction, words_consumed, error_message);
  }
  if (words.size() >= 2 && IsFlatInstructionWord(word)) {
    return DecodeFlat(words, instruction, words_consumed, error_message);
  }
  if (words.size() >= 2 && IsDsInstructionWord(word)) {
    return DecodeDs(words, instruction, words_consumed, error_message);
  }
  if (words.size() >= 2 && ExtractBits(word, 26, 6) == kEncSmem) {
    return DecodeSmem(words, instruction, words_consumed, error_message);
  }
  if (words.size() >= 2 && ExtractBits(word, 26, 6) == kEncVop3) {
    return DecodeVop3(words, instruction, words_consumed, error_message);
  }
  if (ExtractBits(word, 23, 9) == kEncSopp) {
    *words_consumed = 1;
    return DecodeSopp(word, instruction, error_message);
  }
  if (ExtractBits(word, 23, 9) == kEncSopc) {
    return DecodeSopc(words, instruction, words_consumed, error_message);
  }
  if (ExtractBits(word, 23, 9) == kEncSop1) {
    return DecodeSop1(words, instruction, words_consumed, error_message);
  }
  if (ExtractBits(word, 28, 4) == kEncSopk) {
    return DecodeSopk(words, instruction, words_consumed, error_message);
  }
  if (ExtractBits(word, 30, 2) == kEncSop2) {
    return DecodeSop2(words, instruction, words_consumed, error_message);
  }
  if (ExtractBits(word, 25, 7) == kEncVop1) {
    return DecodeVop1(words, instruction, words_consumed, error_message);
  }
  if (ExtractBits(word, 25, 7) == kEncVopc) {
    return DecodeVopc(words, instruction, words_consumed, error_message);
  }
  if (ExtractBits(word, 31, 1) == kEncVop2) {
    return DecodeVop2(words, instruction, words_consumed, error_message);
  }

  if (error_message != nullptr) {
    *error_message = "unsupported or unknown gfx950 binary encoding";
  }
  return false;
}

bool Gfx950BinaryDecoder::DecodeProgram(std::span<const std::uint32_t> words,
                                        std::vector<DecodedInstruction>* program,
                                        std::string* error_message) const {
  if (program == nullptr) {
    if (error_message != nullptr) {
      *error_message = "decoded program output must not be null";
    }
    return false;
  }

  program->clear();
  std::size_t offset = 0;
  while (offset < words.size()) {
    DecodedInstruction instruction;
    std::size_t words_consumed = 0;
    if (!DecodeInstruction(words.subspan(offset), &instruction, &words_consumed,
                           error_message)) {
      return false;
    }
    if (words_consumed == 0) {
      if (error_message != nullptr) {
        *error_message = "decoder did not consume any words";
      }
      return false;
    }
    program->push_back(instruction);
    offset += words_consumed;
  }
  return true;
}

const char* Gfx950BinaryDecoder::FindInstructionName(const char* encoding_name,
                                                     std::uint32_t opcode) const {
  const auto instructions = GetGfx950InstructionSpecs();
  for (const InstructionSpec& instruction : instructions) {
    for (const InstructionEncodingSpec& encoding : GetEncodings(instruction)) {
      if (encoding.encoding_name == encoding_name &&
          encoding.encoding_condition == "default" && encoding.opcode == opcode) {
        return instruction.instruction_name.data();
      }
    }
  }
  return nullptr;
}

bool Gfx950BinaryDecoder::DecodeSopp(std::uint32_t word,
                                     DecodedInstruction* instruction,
                                     std::string* error_message) const {
  const std::uint32_t opcode = ExtractBits(word, 16, 7);
  const char* instruction_name = FindInstructionName("ENC_SOPP", opcode);
  if (instruction_name == nullptr) {
    if (error_message != nullptr) {
      *error_message = "unknown SOPP opcode";
    }
    return false;
  }

  if (instruction_name == std::string_view("S_ENDPGM") ||
      instruction_name == std::string_view("S_BARRIER")) {
    *instruction = DecodedInstruction::Nullary(instruction_name);
    return true;
  }

  if (instruction_name == std::string_view("S_BRANCH") ||
      instruction_name == std::string_view("S_CBRANCH_SCC0") ||
      instruction_name == std::string_view("S_CBRANCH_SCC1") ||
      instruction_name == std::string_view("S_CBRANCH_VCCZ") ||
      instruction_name == std::string_view("S_CBRANCH_VCCNZ") ||
      instruction_name == std::string_view("S_CBRANCH_EXECZ") ||
      instruction_name == std::string_view("S_CBRANCH_EXECNZ")) {
    *instruction = DecodedInstruction::OneOperand(
        instruction_name,
        InstructionOperand::Imm32(
            static_cast<std::uint32_t>(SignExtend16(ExtractBits(word, 0, 16)))));
    return true;
  }

  if (error_message != nullptr) {
    *error_message = "unsupported SOPP instruction";
  }
  return false;
}

bool Gfx950BinaryDecoder::DecodeSmem(std::span<const std::uint32_t> words,
                                     DecodedInstruction* instruction,
                                     std::size_t* words_consumed,
                                     std::string* error_message) const {
  if (words.size() < 2) {
    if (error_message != nullptr) {
      *error_message = "smem instruction requires two dwords";
    }
    return false;
  }

  const std::uint64_t instruction_word =
      static_cast<std::uint64_t>(words[0]) |
      (static_cast<std::uint64_t>(words[1]) << 32);
  const std::uint32_t opcode =
      static_cast<std::uint32_t>(ExtractBits(instruction_word, 18, 8));
  const char* instruction_name = FindInstructionName("ENC_SMEM", opcode);
  if (instruction_name == nullptr) {
    if (error_message != nullptr) {
      *error_message = "unknown SMEM opcode";
    }
    return false;
  }

  if (std::string_view(instruction_name) != "S_LOAD_DWORD" &&
      std::string_view(instruction_name) != "S_LOAD_DWORDX2" &&
      std::string_view(instruction_name) != "S_STORE_DWORD") {
    if (error_message != nullptr) {
      *error_message = "unsupported smem opcode";
    }
    return false;
  }

  InstructionOperand sdata;
  if (!DecodeScalarDestination(
          static_cast<std::uint32_t>(ExtractBits(instruction_word, 6, 7)), &sdata,
          error_message)) {
    return false;
  }

  InstructionOperand sbase;
  if (!DecodeSmemBase(
          static_cast<std::uint32_t>(ExtractBits(instruction_word, 0, 6)), &sbase,
          error_message)) {
    return false;
  }

  InstructionOperand offset;
  if (!DecodeSmemOffset(instruction_word, &offset, error_message)) {
    return false;
  }

  *instruction = DecodedInstruction::ThreeOperand(instruction_name, sdata, sbase,
                                                  offset);
  *words_consumed = 2;
  return true;
}

bool Gfx950BinaryDecoder::DecodeFlat(std::span<const std::uint32_t> words,
                                     DecodedInstruction* instruction,
                                     std::size_t* words_consumed,
                                     std::string* error_message) const {
  if (words.size() < 2) {
    if (error_message != nullptr) {
      *error_message = "flat instruction requires two dwords";
    }
    return false;
  }

  const std::uint64_t instruction_word =
      static_cast<std::uint64_t>(words[0]) |
      (static_cast<std::uint64_t>(words[1]) << 32);
  if (ExtractBits(instruction_word, 55, 1) != 0) {
    if (error_message != nullptr) {
      *error_message = "accvgpr flat operands are not implemented";
    }
    return false;
  }

  const std::uint32_t opcode =
      static_cast<std::uint32_t>(ExtractBits(instruction_word, 18, 7));
  const char* instruction_name = FindInstructionName("ENC_FLAT", opcode);
  if (instruction_name == nullptr) {
    if (error_message != nullptr) {
      *error_message = "unknown FLAT opcode";
    }
    return false;
  }

  const std::string_view opcode_name(instruction_name);
  if (!IsSupportedFlatVectorMemoryOpcode(opcode_name) &&
      !opcode_name.starts_with("FLAT_ATOMIC_")) {
    if (error_message != nullptr) {
      *error_message = "unsupported flat opcode";
    }
    return false;
  }

  InstructionOperand addr;
  if (!DecodeFlatAddress(
          static_cast<std::uint32_t>(ExtractBits(instruction_word, 32, 8)), &addr,
          error_message)) {
    return false;
  }
  const InstructionOperand offset = InstructionOperand::Imm32(
      static_cast<std::uint32_t>(ExtractBits(instruction_word, 0, 12)));
  const bool return_prior_value = ExtractBits(instruction_word, 16, 1) != 0;

  if (opcode_name.starts_with("FLAT_LOAD_")) {
    InstructionOperand dst;
    if (!DecodeVectorDestination(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 56, 8)),
            &dst, error_message)) {
      return false;
    }
    *instruction = DecodedInstruction::ThreeOperand(instruction_name, dst, addr,
                                                    offset);
  } else {
    InstructionOperand data;
    if (!DecodeVectorRegisterSource(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 40, 8)),
            &data, error_message)) {
      return false;
    }
    if (opcode_name.starts_with("FLAT_STORE_")) {
      *instruction = DecodedInstruction::ThreeOperand(instruction_name, addr,
                                                      data, offset);
    } else if (return_prior_value) {
      InstructionOperand dst;
      if (!DecodeVectorDestination(
              static_cast<std::uint32_t>(ExtractBits(instruction_word, 56, 8)),
              &dst, error_message)) {
        return false;
      }
      *instruction = DecodedInstruction::FourOperand(instruction_name, dst, addr,
                                                     data, offset);
    } else {
      *instruction =
          DecodedInstruction::ThreeOperand(instruction_name, addr, data, offset);
    }
  }

  *words_consumed = 2;
  return true;
}

bool Gfx950BinaryDecoder::DecodeDs(std::span<const std::uint32_t> words,
                                   DecodedInstruction* instruction,
                                   std::size_t* words_consumed,
                                   std::string* error_message) const {
  if (words.size() < 2) {
    if (error_message != nullptr) {
      *error_message = "ds instruction requires two dwords";
    }
    return false;
  }

  const std::uint64_t instruction_word =
      static_cast<std::uint64_t>(words[0]) |
      (static_cast<std::uint64_t>(words[1]) << 32);
  if (ExtractBits(instruction_word, 25, 1) != 0) {
    if (error_message != nullptr) {
      *error_message = "accvgpr ds operands are not implemented";
    }
    return false;
  }
  if (ExtractBits(instruction_word, 16, 1) != 0) {
    if (error_message != nullptr) {
      *error_message = "gds ds operations are not implemented";
    }
    return false;
  }

  const std::uint32_t opcode =
      static_cast<std::uint32_t>(ExtractBits(instruction_word, 17, 8));
  const char* instruction_name = FindInstructionName("ENC_DS", opcode);
  if (instruction_name == nullptr) {
    if (error_message != nullptr) {
      *error_message = "unknown DS opcode";
    }
    return false;
  }

  const std::string_view opcode_name(instruction_name);
  if (!IsSupportedDsOpcode(opcode_name)) {
    if (error_message != nullptr) {
      *error_message = "unsupported ds opcode";
    }
    return false;
  }

  const std::uint32_t raw_offset0 =
      static_cast<std::uint32_t>(ExtractBits(instruction_word, 0, 8));
  const std::uint32_t raw_offset1 =
      static_cast<std::uint32_t>(ExtractBits(instruction_word, 8, 8));
  const InstructionOperand offset0 = InstructionOperand::Imm32(raw_offset0);
  const InstructionOperand offset1 = InstructionOperand::Imm32(raw_offset1);
  const InstructionOperand combined_offset = InstructionOperand::Imm32(
      raw_offset0 | (raw_offset1 << 8));
  if (!IsDsPairWriteOpcode(opcode_name) && !IsDsPairReadOpcode(opcode_name) &&
      !IsDsPairReturnOpcode(opcode_name) &&
      !IsDsWaveCounterOpcode(opcode_name) &&
      !IsDsLaneRoutingOpcode(opcode_name) && offset1.imm32 != 0) {
    if (error_message != nullptr) {
      *error_message = "offset1 ds forms are not implemented";
    }
    return false;
  }

  if (opcode_name == "DS_NOP") {
    *instruction = DecodedInstruction::Nullary(instruction_name);
    *words_consumed = 2;
    return true;
  }

  if (IsDsWaveCounterOpcode(opcode_name)) {
    InstructionOperand dst;
    if (!DecodeVectorDestination(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 56, 8)),
            &dst, error_message)) {
      return false;
    }
    *instruction = DecodedInstruction::TwoOperand(instruction_name, dst,
                                                  combined_offset);
    *words_consumed = 2;
    return true;
  }

  InstructionOperand addr;
  if (!DecodeVectorRegisterSource(
          static_cast<std::uint32_t>(ExtractBits(instruction_word, 32, 8)), &addr,
          error_message)) {
    return false;
  }

  if (IsDsPairWriteOpcode(opcode_name)) {
    InstructionOperand data0;
    if (!DecodeVectorRegisterSource(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 40, 8)),
            &data0, error_message)) {
      return false;
    }
    InstructionOperand data1;
    if (!DecodeVectorRegisterSource(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 48, 8)),
            &data1, error_message)) {
      return false;
    }
    *instruction = DecodedInstruction::FiveOperand(instruction_name, addr, data0,
                                                   data1, offset0, offset1);
  } else if (IsDsPairReadOpcode(opcode_name)) {
    InstructionOperand dst;
    if (!DecodeVectorDestination(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 56, 8)),
            &dst, error_message)) {
      return false;
    }
    *instruction = DecodedInstruction::FourOperand(instruction_name, dst, addr,
                                                   offset0, offset1);
  } else if (IsDsPairReturnOpcode(opcode_name)) {
    InstructionOperand dst;
    if (!DecodeVectorDestination(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 56, 8)),
            &dst, error_message)) {
      return false;
    }
    InstructionOperand data0;
    if (!DecodeVectorRegisterSource(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 40, 8)),
            &data0, error_message)) {
      return false;
    }
    InstructionOperand data1;
    if (!DecodeVectorRegisterSource(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 48, 8)),
            &data1, error_message)) {
      return false;
    }
    *instruction = DecodedInstruction::SixOperand(instruction_name, dst, addr,
                                                  data0, data1, offset0,
                                                  offset1);
  } else if (IsDsSwizzleOpcode(opcode_name)) {
    InstructionOperand dst;
    if (!DecodeVectorDestination(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 56, 8)),
            &dst, error_message)) {
      return false;
    }
    *instruction = DecodedInstruction::ThreeOperand(instruction_name, dst, addr,
                                                    combined_offset);
  } else if (IsDsPermuteOpcode(opcode_name)) {
    InstructionOperand dst;
    if (!DecodeVectorDestination(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 56, 8)),
            &dst, error_message)) {
      return false;
    }
    InstructionOperand data0;
    if (!DecodeVectorRegisterSource(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 40, 8)),
            &data0, error_message)) {
      return false;
    }
    *instruction = DecodedInstruction::FourOperand(instruction_name, dst, addr,
                                                   data0, combined_offset);
  } else if (IsDsDualDataReturnOpcode(opcode_name)) {
    InstructionOperand dst;
    if (!DecodeVectorDestination(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 56, 8)),
            &dst, error_message)) {
      return false;
    }
    InstructionOperand data0;
    if (!DecodeVectorRegisterSource(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 40, 8)),
            &data0, error_message)) {
      return false;
    }
    InstructionOperand data1;
    if (!DecodeVectorRegisterSource(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 48, 8)),
            &data1, error_message)) {
      return false;
    }
    *instruction = DecodedInstruction::FiveOperand(instruction_name, dst, addr,
                                                   data0, data1, offset0);
  } else if (IsDsAddTidReadOpcode(opcode_name)) {
    InstructionOperand dst;
    if (!DecodeVectorDestination(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 56, 8)),
            &dst, error_message)) {
      return false;
    }
    *instruction = DecodedInstruction::TwoOperand(instruction_name, dst, offset0);
  } else if (IsDsAddTidWriteOpcode(opcode_name)) {
    InstructionOperand data0;
    if (!DecodeVectorRegisterSource(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 40, 8)),
            &data0, error_message)) {
      return false;
    }
    *instruction = DecodedInstruction::TwoOperand(instruction_name, data0, offset0);
  } else if (IsDsDirectReadOpcode(opcode_name)) {
    InstructionOperand dst;
    if (!DecodeVectorDestination(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 56, 8)),
            &dst, error_message)) {
      return false;
    }
    *instruction = DecodedInstruction::ThreeOperand(instruction_name, dst, addr,
                                                    offset0);
  } else if (IsDsReturnOpcode(opcode_name)) {
    InstructionOperand dst;
    if (!DecodeVectorDestination(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 56, 8)),
            &dst, error_message)) {
      return false;
    }
    InstructionOperand data;
    if (!DecodeVectorRegisterSource(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 40, 8)),
            &data, error_message)) {
      return false;
    }
    *instruction = DecodedInstruction::FourOperand(instruction_name, dst, addr,
                                                   data, offset0);
  } else if (IsDsDualDataOpcode(opcode_name)) {
    InstructionOperand data0;
    if (!DecodeVectorRegisterSource(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 40, 8)),
            &data0, error_message)) {
      return false;
    }
    InstructionOperand data1;
    if (!DecodeVectorRegisterSource(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 48, 8)),
            &data1, error_message)) {
      return false;
    }
    *instruction = DecodedInstruction::FourOperand(instruction_name, addr, data0,
                                                   data1, offset0);
  } else {
    InstructionOperand data;
    if (!DecodeVectorRegisterSource(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 40, 8)),
            &data, error_message)) {
      return false;
    }
    *instruction = DecodedInstruction::ThreeOperand(instruction_name, addr, data,
                                                    offset0);
  }

  *words_consumed = 2;
  return true;
}

bool Gfx950BinaryDecoder::DecodeFlatGlobal(
    std::span<const std::uint32_t> words,
    DecodedInstruction* instruction,
    std::size_t* words_consumed,
    std::string* error_message) const {
  if (words.size() < 2) {
    if (error_message != nullptr) {
      *error_message = "global instruction requires two dwords";
    }
    return false;
  }

  const std::uint64_t instruction_word =
      static_cast<std::uint64_t>(words[0]) |
      (static_cast<std::uint64_t>(words[1]) << 32);
  if (ExtractBits(instruction_word, 55, 1) != 0) {
    if (error_message != nullptr) {
      *error_message = "accvgpr global operands are not implemented";
    }
    return false;
  }

  const std::uint32_t opcode =
      static_cast<std::uint32_t>(ExtractBits(instruction_word, 18, 7));
  const char* instruction_name = FindInstructionName("ENC_FLAT_GLBL", opcode);
  if (instruction_name == nullptr) {
    if (error_message != nullptr) {
      *error_message = "unknown GLOBAL opcode";
    }
    return false;
  }

  const std::string_view opcode_name(instruction_name);
  if (!IsSupportedGlobalVectorMemoryOpcode(opcode_name) &&
      !opcode_name.starts_with("GLOBAL_ATOMIC_")) {
    if (error_message != nullptr) {
      *error_message = "unsupported global opcode";
    }
    return false;
  }

  InstructionOperand addr;
  if (!DecodeFlatAddress(
          static_cast<std::uint32_t>(ExtractBits(instruction_word, 32, 8)), &addr,
          error_message)) {
    return false;
  }

  InstructionOperand saddr;
  if (!DecodeFlatGlobalBase(
          static_cast<std::uint32_t>(ExtractBits(instruction_word, 48, 7)),
          &saddr, error_message)) {
    return false;
  }
  const InstructionOperand offset = InstructionOperand::Imm32(
      static_cast<std::uint32_t>(SignExtend13(static_cast<std::uint32_t>(
          ExtractBits(instruction_word, 0, 13)))));
  const bool return_prior_value = ExtractBits(instruction_word, 16, 1) != 0;

  if (opcode_name.starts_with("GLOBAL_LOAD_")) {
    InstructionOperand dst;
    if (!DecodeVectorDestination(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 56, 8)),
            &dst, error_message)) {
      return false;
    }
    *instruction = DecodedInstruction::FourOperand(instruction_name, dst, addr,
                                                   saddr, offset);
  } else if (opcode_name.starts_with("GLOBAL_STORE_")) {
    InstructionOperand data;
    if (!DecodeVectorRegisterSource(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 40, 8)),
            &data, error_message)) {
      return false;
    }
    *instruction = DecodedInstruction::FourOperand(instruction_name, addr, data,
                                                   saddr, offset);
  } else {
    InstructionOperand data;
    if (!DecodeVectorRegisterSource(
            static_cast<std::uint32_t>(ExtractBits(instruction_word, 40, 8)),
            &data, error_message)) {
      return false;
    }
    if (return_prior_value) {
      InstructionOperand dst;
      if (!DecodeVectorDestination(
              static_cast<std::uint32_t>(ExtractBits(instruction_word, 56, 8)),
              &dst, error_message)) {
        return false;
      }
      *instruction = DecodedInstruction::FiveOperand(instruction_name, dst, addr,
                                                     data, saddr, offset);
    } else {
      *instruction =
          DecodedInstruction::FourOperand(instruction_name, addr, data, saddr,
                                          offset);
    }
  }

  *words_consumed = 2;
  return true;
}

bool Gfx950BinaryDecoder::DecodeSopc(std::span<const std::uint32_t> words,
                                     DecodedInstruction* instruction,
                                     std::size_t* words_consumed,
                                     std::string* error_message) const {
  const std::uint32_t word = words.front();
  const std::uint32_t opcode = ExtractBits(word, 16, 7);
  const char* instruction_name = FindInstructionName("ENC_SOPC", opcode);
  if (instruction_name == nullptr) {
    if (error_message != nullptr) {
      *error_message = "unknown SOPC opcode";
    }
    return false;
  }

  std::size_t literal_words_consumed = 0;
  InstructionOperand src0;
  if (!DecodeScalarSource(ExtractBits(word, 0, 8), words.subspan(1),
                          &literal_words_consumed, &src0, error_message)) {
    return false;
  }

  std::size_t src1_literal_words_consumed = 0;
  InstructionOperand src1;
  if (!DecodeScalarSource(ExtractBits(word, 8, 8),
                          words.subspan(1 + literal_words_consumed),
                          &src1_literal_words_consumed, &src1, error_message)) {
    return false;
  }

  *instruction = DecodedInstruction::TwoOperand(instruction_name, src0, src1);
  *words_consumed = 1 + literal_words_consumed + src1_literal_words_consumed;
  return true;
}

bool Gfx950BinaryDecoder::DecodeSopk(std::span<const std::uint32_t> words,
                                     DecodedInstruction* instruction,
                                     std::size_t* words_consumed,
                                     std::string* error_message) const {
  const std::uint32_t word = words.front();
  const std::uint32_t opcode = ExtractBits(word, 23, 5);
  const char* instruction_name = FindInstructionName("ENC_SOPK", opcode);
  if (instruction_name == nullptr) {
    if (error_message != nullptr) {
      *error_message = "unknown SOPK opcode";
    }
    return false;
  }

  InstructionOperand dst;
  if (!DecodeScalarDestination(ExtractBits(word, 16, 7), &dst, error_message)) {
    return false;
  }

  const std::uint32_t raw_imm16 = ExtractBits(word, 0, 16);
  const std::uint32_t signed_imm16 =
      static_cast<std::uint32_t>(SignExtend16(raw_imm16));

  if (instruction_name == std::string_view("S_MOVK_I32")) {
    *instruction =
        DecodedInstruction::Unary("S_MOV_B32", dst, InstructionOperand::Imm32(signed_imm16));
  } else if (instruction_name == std::string_view("S_CMOVK_I32")) {
    *instruction = DecodedInstruction::Unary("S_CMOV_B32", dst,
                                             InstructionOperand::Imm32(signed_imm16));
  } else if (instruction_name == std::string_view("S_ADDK_I32")) {
    *instruction = DecodedInstruction::Binary("S_ADD_U32", dst, dst,
                                              InstructionOperand::Imm32(signed_imm16));
  } else if (instruction_name == std::string_view("S_MULK_I32")) {
    *instruction = DecodedInstruction::Binary("S_MUL_I32", dst, dst,
                                              InstructionOperand::Imm32(signed_imm16));
  } else if (instruction_name == std::string_view("S_CMPK_EQ_I32")) {
    *instruction = DecodedInstruction::TwoOperand(
        "S_CMP_EQ_I32", dst, InstructionOperand::Imm32(signed_imm16));
  } else if (instruction_name == std::string_view("S_CMPK_LG_I32")) {
    *instruction = DecodedInstruction::TwoOperand(
        "S_CMP_LG_I32", dst, InstructionOperand::Imm32(signed_imm16));
  } else if (instruction_name == std::string_view("S_CMPK_GT_I32")) {
    *instruction = DecodedInstruction::TwoOperand(
        "S_CMP_GT_I32", dst, InstructionOperand::Imm32(signed_imm16));
  } else if (instruction_name == std::string_view("S_CMPK_GE_I32")) {
    *instruction = DecodedInstruction::TwoOperand(
        "S_CMP_GE_I32", dst, InstructionOperand::Imm32(signed_imm16));
  } else if (instruction_name == std::string_view("S_CMPK_LT_I32")) {
    *instruction = DecodedInstruction::TwoOperand(
        "S_CMP_LT_I32", dst, InstructionOperand::Imm32(signed_imm16));
  } else if (instruction_name == std::string_view("S_CMPK_LE_I32")) {
    *instruction = DecodedInstruction::TwoOperand(
        "S_CMP_LE_I32", dst, InstructionOperand::Imm32(signed_imm16));
  } else if (instruction_name == std::string_view("S_CMPK_EQ_U32")) {
    *instruction = DecodedInstruction::TwoOperand(
        "S_CMP_EQ_U32", dst, InstructionOperand::Imm32(raw_imm16));
  } else if (instruction_name == std::string_view("S_CMPK_LG_U32")) {
    *instruction = DecodedInstruction::TwoOperand(
        "S_CMP_LG_U32", dst, InstructionOperand::Imm32(raw_imm16));
  } else if (instruction_name == std::string_view("S_CMPK_GT_U32")) {
    *instruction = DecodedInstruction::TwoOperand(
        "S_CMP_GT_U32", dst, InstructionOperand::Imm32(raw_imm16));
  } else if (instruction_name == std::string_view("S_CMPK_GE_U32")) {
    *instruction = DecodedInstruction::TwoOperand(
        "S_CMP_GE_U32", dst, InstructionOperand::Imm32(raw_imm16));
  } else if (instruction_name == std::string_view("S_CMPK_LT_U32")) {
    *instruction = DecodedInstruction::TwoOperand(
        "S_CMP_LT_U32", dst, InstructionOperand::Imm32(raw_imm16));
  } else if (instruction_name == std::string_view("S_CMPK_LE_U32")) {
    *instruction = DecodedInstruction::TwoOperand(
        "S_CMP_LE_U32", dst, InstructionOperand::Imm32(raw_imm16));
  } else {
    if (error_message != nullptr) {
      *error_message = "unsupported SOPK instruction";
    }
    return false;
  }

  *words_consumed = 1;
  return true;
}

bool Gfx950BinaryDecoder::DecodeSop1(std::span<const std::uint32_t> words,
                                     DecodedInstruction* instruction,
                                     std::size_t* words_consumed,
                                     std::string* error_message) const {
  const std::uint32_t word = words.front();
  const std::uint32_t opcode = ExtractBits(word, 8, 8);
  const char* instruction_name = FindInstructionName("ENC_SOP1", opcode);
  if (instruction_name == nullptr) {
    if (error_message != nullptr) {
      *error_message = "unknown SOP1 opcode";
    }
    return false;
  }

  InstructionOperand dst;
  if (!DecodeScalarDestination(ExtractBits(word, 16, 7), &dst, error_message)) {
    return false;
  }

  std::size_t literal_words_consumed = 0;
  InstructionOperand src0;
  if (!DecodeScalarSource(ExtractBits(word, 0, 8), words.subspan(1),
                          &literal_words_consumed, &src0, error_message)) {
    return false;
  }

  *instruction = DecodedInstruction::Unary(instruction_name, dst, src0);
  *words_consumed = 1 + literal_words_consumed;
  return true;
}

bool Gfx950BinaryDecoder::DecodeSop2(std::span<const std::uint32_t> words,
                                     DecodedInstruction* instruction,
                                     std::size_t* words_consumed,
                                     std::string* error_message) const {
  const std::uint32_t word = words.front();
  const std::uint32_t opcode = ExtractBits(word, 23, 7);
  const char* instruction_name = FindInstructionName("ENC_SOP2", opcode);
  if (instruction_name == nullptr) {
    if (error_message != nullptr) {
      *error_message = "unknown SOP2 opcode";
    }
    return false;
  }

  InstructionOperand dst;
  if (!DecodeScalarDestination(ExtractBits(word, 16, 7), &dst, error_message)) {
    return false;
  }

  std::size_t literal_words_consumed = 0;
  InstructionOperand src0;
  if (!DecodeScalarSource(ExtractBits(word, 0, 8), words.subspan(1),
                          &literal_words_consumed, &src0, error_message)) {
    return false;
  }

  std::size_t src1_literal_words_consumed = 0;
  InstructionOperand src1;
  if (!DecodeScalarSource(ExtractBits(word, 8, 8),
                          words.subspan(1 + literal_words_consumed),
                          &src1_literal_words_consumed, &src1, error_message)) {
    return false;
  }

  *instruction = DecodedInstruction::Binary(instruction_name, dst, src0, src1);
  *words_consumed = 1 + literal_words_consumed + src1_literal_words_consumed;
  return true;
}

bool Gfx950BinaryDecoder::DecodeVop1(std::span<const std::uint32_t> words,
                                     DecodedInstruction* instruction,
                                     std::size_t* words_consumed,
                                     std::string* error_message) const {
  const std::uint32_t word = words.front();
  const std::uint32_t opcode = ExtractBits(word, 9, 8);
  const char* instruction_name = FindInstructionName("ENC_VOP1", opcode);
  if (instruction_name == nullptr) {
    if (error_message != nullptr) {
      *error_message = "unknown VOP1 opcode";
    }
    return false;
  }

  if (instruction_name == std::string_view("V_NOP")) {
    *instruction = DecodedInstruction::Nullary(instruction_name);
    *words_consumed = 1;
    return true;
  }

  InstructionOperand dst;
  if (instruction_name == std::string_view("V_READFIRSTLANE_B32")) {
    if (!DecodeScalarDestination(ExtractBits(word, 17, 8), &dst, error_message)) {
      return false;
    }
  } else if (!DecodeVectorDestination(ExtractBits(word, 17, 8), &dst,
                                      error_message)) {
    return false;
  }

  std::size_t literal_words_consumed = 0;
  InstructionOperand src0;
  if (!DecodeVectorSource(ExtractBits(word, 0, 9), words.subspan(1),
                          &literal_words_consumed, &src0, error_message)) {
    return false;
  }

  *instruction = DecodedInstruction::Unary(instruction_name, dst, src0);
  *words_consumed = 1 + literal_words_consumed;
  return true;
}

bool Gfx950BinaryDecoder::DecodeVop2(std::span<const std::uint32_t> words,
                                     DecodedInstruction* instruction,
                                     std::size_t* words_consumed,
                                     std::string* error_message) const {
  const std::uint32_t word = words.front();
  const std::uint32_t opcode = ExtractBits(word, 25, 6);
  const char* instruction_name = FindInstructionName("ENC_VOP2", opcode);
  if (instruction_name == nullptr) {
    if (error_message != nullptr) {
      *error_message = "unknown VOP2 opcode";
    }
    return false;
  }

  InstructionOperand dst;
  if (!DecodeVectorDestination(ExtractBits(word, 17, 8), &dst, error_message)) {
    return false;
  }

  std::size_t literal_words_consumed = 0;
  InstructionOperand src0;
  if (!DecodeVectorSource(ExtractBits(word, 0, 9), words.subspan(1),
                          &literal_words_consumed, &src0, error_message)) {
    return false;
  }

  InstructionOperand src1;
  if (!DecodeVectorRegisterSource(ExtractBits(word, 9, 8), &src1, error_message)) {
    return false;
  }

  if (IsVectorCarryOutBinaryOpcode(instruction_name)) {
    *instruction = DecodedInstruction::FourOperand(
        instruction_name, dst, InstructionOperand::Sgpr(kImplicitVccPairSgprIndex),
        src0, src1);
  } else if (IsVectorCarryInBinaryOpcode(instruction_name)) {
    *instruction = DecodedInstruction::FiveOperand(
        instruction_name, dst, InstructionOperand::Sgpr(kImplicitVccPairSgprIndex),
        src0, src1, InstructionOperand::Sgpr(kImplicitVccPairSgprIndex));
  } else {
    *instruction = DecodedInstruction::Binary(instruction_name, dst, src0, src1);
  }
  *words_consumed = 1 + literal_words_consumed;
  return true;
}

bool Gfx950BinaryDecoder::DecodeVopc(std::span<const std::uint32_t> words,
                                     DecodedInstruction* instruction,
                                     std::size_t* words_consumed,
                                     std::string* error_message) const {
  const std::uint32_t word = words.front();
  const std::uint32_t opcode = ExtractBits(word, 17, 8);
  const char* instruction_name = FindInstructionName("ENC_VOPC", opcode);
  if (instruction_name == nullptr) {
    if (error_message != nullptr) {
      *error_message = "unknown VOPC opcode";
    }
    return false;
  }

  if (!IsSupportedVectorCompareOpcode(instruction_name)) {
    if (error_message != nullptr) {
      *error_message = "unsupported VOPC opcode";
    }
    return false;
  }

  std::size_t literal_words_consumed = 0;
  InstructionOperand src0;
  if (!DecodeVectorSource(ExtractBits(word, 0, 9), words.subspan(1),
                          &literal_words_consumed, &src0, error_message)) {
    return false;
  }

  InstructionOperand src1;
  if (!DecodeVectorRegisterSource(ExtractBits(word, 9, 8), &src1,
                                  error_message)) {
    return false;
  }

  *instruction =
      DecodedInstruction::Binary(instruction_name,
                                 InstructionOperand::Sgpr(
                                     kImplicitVccPairSgprIndex),
                                 src0, src1);
  *words_consumed = 1 + literal_words_consumed;
  return true;
}

bool Gfx950BinaryDecoder::DecodeVop3(std::span<const std::uint32_t> words,
                                     DecodedInstruction* instruction,
                                     std::size_t* words_consumed,
                                     std::string* error_message) const {
  if (words.size() < 2) {
    if (error_message != nullptr) {
      *error_message = "vop3 instruction requires two dwords";
    }
    return false;
  }

  const std::uint64_t instruction_word =
      static_cast<std::uint64_t>(words[0]) |
      (static_cast<std::uint64_t>(words[1]) << 32);
  const std::uint32_t opcode =
      static_cast<std::uint32_t>(ExtractBits(instruction_word, 16, 10));
  const char* instruction_name = FindInstructionName("ENC_VOP3", opcode);
  bool is_vop3_sdst = false;
  if (instruction_name == nullptr) {
    instruction_name = FindInstructionName("VOP3_SDST_ENC", opcode);
    is_vop3_sdst = instruction_name != nullptr;
  }
  if (instruction_name == nullptr) {
    if (error_message != nullptr) {
      *error_message = "unknown VOP3 opcode";
    }
    return false;
  }

  if ((!is_vop3_sdst &&
       (ExtractBits(instruction_word, 8, 3) != 0 ||
        ExtractBits(instruction_word, 11, 4) != 0)) ||
      ExtractBits(instruction_word, 15, 1) != 0 ||
      ExtractBits(instruction_word, 59, 2) != 0 ||
      ExtractBits(instruction_word, 61, 3) != 0) {
    if (error_message != nullptr) {
      *error_message = "VOP3 modifiers are not implemented";
    }
    return false;
  }

  std::size_t src0_literal_words_consumed = 0;
  InstructionOperand src0;
  const bool is_v_readlane =
      instruction_name == std::string_view("V_READLANE_B32");
  const bool is_v_writelane =
      instruction_name == std::string_view("V_WRITELANE_B32");
  const bool is_v_readfirstlane =
      instruction_name == std::string_view("V_READFIRSTLANE_B32");

  const std::uint32_t raw_src0 =
      static_cast<std::uint32_t>(ExtractBits(instruction_word, 32, 9));
  if ((is_v_writelane
           ? !DecodeScalarSource(raw_src0, std::span<const std::uint32_t>(),
                                 &src0_literal_words_consumed, &src0,
                                 error_message)
           : !DecodeVectorSource(raw_src0, std::span<const std::uint32_t>(),
                                 &src0_literal_words_consumed, &src0,
                                 error_message))) {
    return false;
  }

  std::size_t src1_literal_words_consumed = 0;
  InstructionOperand src1;
  const std::uint32_t raw_src1 =
      static_cast<std::uint32_t>(ExtractBits(instruction_word, 41, 9));
  if (((is_v_readlane || is_v_writelane)
           ? !DecodeScalarSource(raw_src1, std::span<const std::uint32_t>(),
                                 &src1_literal_words_consumed, &src1,
                                 error_message)
           : !DecodeVectorSource(raw_src1, std::span<const std::uint32_t>(),
                                 &src1_literal_words_consumed, &src1,
                                 error_message))) {
    return false;
  }

  const bool is_vector_compare =
      IsSupportedVectorCompareOpcode(instruction_name);

  InstructionOperand dst;
  if (is_vector_compare || is_v_readlane || is_v_readfirstlane) {
    if (!DecodeScalarDestination(static_cast<std::uint32_t>(
                                     ExtractBits(instruction_word, 0, 8)),
                                 &dst, error_message)) {
      return false;
    }
  } else if (!DecodeVectorDestination(static_cast<std::uint32_t>(
                                          ExtractBits(instruction_word, 0, 8)),
                                      &dst, error_message)) {
    return false;
  }

  if (is_vector_compare) {
    if (ExtractBits(instruction_word, 50, 9) != 0) {
      if (error_message != nullptr) {
        *error_message = "three-source VOP3 compare forms are not implemented";
      }
      return false;
    }
    *instruction = DecodedInstruction::Binary(instruction_name, dst, src0, src1);
    *words_consumed = 2;
    return true;
  }

  if (instruction_name == std::string_view("V_CNDMASK_B32")) {
    if (src0.kind != OperandKind::kSgpr ||
        src0.index != kImplicitVccPairSgprIndex) {
      if (error_message != nullptr) {
        *error_message = "unsupported VOP3 cndmask condition source";
      }
      return false;
    }
    std::size_t src2_literal_words_consumed = 0;
    InstructionOperand src2;
    if (!DecodeVectorSource(static_cast<std::uint32_t>(
                                ExtractBits(instruction_word, 50, 9)),
                            std::span<const std::uint32_t>(),
                            &src2_literal_words_consumed, &src2, error_message)) {
      return false;
    }
    *instruction =
        DecodedInstruction::Binary(instruction_name, dst, src2, src1);
    *words_consumed = 2;
    return true;
  }

  if (IsSupportedPromotedVop3UnaryOpcode(instruction_name)) {
    if (ExtractBits(instruction_word, 41, 9) != 0 ||
        ExtractBits(instruction_word, 50, 9) != 0) {
      if (error_message != nullptr) {
        *error_message = "extra VOP3 source operands are not implemented";
      }
      return false;
    }
    *instruction = DecodedInstruction::Unary(instruction_name, dst, src0);
    *words_consumed = 2;
    return true;
  }

  if (IsSupportedPromotedVop3BinaryOpcode(instruction_name)) {
    if (ExtractBits(instruction_word, 50, 9) != 0) {
      if (error_message != nullptr) {
        *error_message = "three-source VOP3 forms are not implemented";
      }
      return false;
    }
    *instruction = DecodedInstruction::Binary(instruction_name, dst, src0, src1);
    *words_consumed = 2;
    return true;
  }

  if (is_v_readlane || is_v_writelane) {
    if (ExtractBits(instruction_word, 50, 9) != 0) {
      if (error_message != nullptr) {
        *error_message = "extra VOP3 source operands are not implemented";
      }
      return false;
    }
    *instruction = DecodedInstruction::Binary(instruction_name, dst, src0, src1);
    *words_consumed = 2;
    return true;
  }

  if (is_vop3_sdst && IsVectorCarryOutBinaryOpcode(instruction_name)) {
    InstructionOperand sdst;
    if (!DecodeScalarDestination(static_cast<std::uint32_t>(
                                     ExtractBits(instruction_word, 8, 7)),
                                 &sdst, error_message)) {
      return false;
    }
    if (sdst.kind != OperandKind::kSgpr || sdst.index >= 127) {
      if (error_message != nullptr) {
        *error_message = "unsupported VOP3 carry destination pair";
      }
      return false;
    }
    if (ExtractBits(instruction_word, 50, 9) != 0) {
      if (error_message != nullptr) {
        *error_message = "unexpected VOP3 carry input operand";
      }
      return false;
    }
    *instruction =
        DecodedInstruction::FourOperand(instruction_name, dst, sdst, src0, src1);
    *words_consumed = 2;
    return true;
  }

  if (is_vop3_sdst && IsVectorCarryInBinaryOpcode(instruction_name)) {
    InstructionOperand sdst;
    if (!DecodeScalarDestination(static_cast<std::uint32_t>(
                                     ExtractBits(instruction_word, 8, 7)),
                                 &sdst, error_message)) {
      return false;
    }
    if (sdst.kind != OperandKind::kSgpr || sdst.index >= 127) {
      if (error_message != nullptr) {
        *error_message = "unsupported VOP3 carry destination pair";
      }
      return false;
    }
    std::size_t src2_literal_words_consumed = 0;
    InstructionOperand src2;
    if (!DecodeScalarSource(static_cast<std::uint32_t>(
                                ExtractBits(instruction_word, 50, 9)),
                            std::span<const std::uint32_t>(),
                            &src2_literal_words_consumed, &src2,
                            error_message)) {
      return false;
    }
    if (src2.kind != OperandKind::kSgpr || src2.index >= 127) {
      if (error_message != nullptr) {
        *error_message = "unsupported VOP3 carry input pair";
      }
      return false;
    }
    *instruction = DecodedInstruction::FiveOperand(instruction_name, dst, sdst,
                                                   src0, src1, src2);
    *words_consumed = 2;
    return true;
  }

  if (is_vop3_sdst &&
      (instruction_name == std::string_view("V_MAD_U64_U32") ||
       instruction_name == std::string_view("V_MAD_I64_I32"))) {
    InstructionOperand sdst;
    if (!DecodeScalarDestination(static_cast<std::uint32_t>(
                                     ExtractBits(instruction_word, 8, 7)),
                                 &sdst, error_message)) {
      return false;
    }
    std::size_t src2_literal_words_consumed = 0;
    InstructionOperand src2;
    if (!DecodeVectorSource(static_cast<std::uint32_t>(
                                ExtractBits(instruction_word, 50, 9)),
                            std::span<const std::uint32_t>(),
                            &src2_literal_words_consumed, &src2, error_message)) {
      return false;
    }
    *instruction = DecodedInstruction::FiveOperand(instruction_name, dst, sdst,
                                                   src0, src1, src2);
    *words_consumed = 2;
    return true;
  }

  if (instruction_name == std::string_view("V_ADD3_U32") ||
      instruction_name == std::string_view("V_FMA_F32") ||
      instruction_name == std::string_view("V_FMA_F64") ||
      instruction_name == std::string_view("V_LSHL_ADD_U32") ||
      instruction_name == std::string_view("V_LSHL_ADD_U64") ||
      instruction_name == std::string_view("V_LERP_U8") ||
      instruction_name == std::string_view("V_PERM_B32") ||
      instruction_name == std::string_view("V_BFE_U32") ||
      instruction_name == std::string_view("V_BFE_I32") ||
      instruction_name == std::string_view("V_BFI_B32") ||
      instruction_name == std::string_view("V_ALIGNBIT_B32") ||
      instruction_name == std::string_view("V_ALIGNBYTE_B32") ||
      instruction_name == std::string_view("V_MIN3_I32") ||
      instruction_name == std::string_view("V_MIN3_U32") ||
      instruction_name == std::string_view("V_MAX3_I32") ||
      instruction_name == std::string_view("V_MAX3_U32") ||
      instruction_name == std::string_view("V_MED3_I32") ||
      instruction_name == std::string_view("V_MED3_U32") ||
      instruction_name == std::string_view("V_SAD_U8") ||
      instruction_name == std::string_view("V_SAD_HI_U8") ||
      instruction_name == std::string_view("V_SAD_U16") ||
      instruction_name == std::string_view("V_SAD_U32") ||
      instruction_name == std::string_view("V_MAD_I32_I24") ||
      instruction_name == std::string_view("V_MAD_U32_U24") ||
      instruction_name == std::string_view("V_ADD_LSHL_U32") ||
      instruction_name == std::string_view("V_LSHL_OR_B32") ||
      instruction_name == std::string_view("V_AND_OR_B32") ||
      instruction_name == std::string_view("V_OR3_B32") ||
      instruction_name == std::string_view("V_XAD_U32")) {
    std::size_t src2_literal_words_consumed = 0;
    InstructionOperand src2;
    if (!DecodeVectorSource(static_cast<std::uint32_t>(
                                ExtractBits(instruction_word, 50, 9)),
                            std::span<const std::uint32_t>(),
                            &src2_literal_words_consumed, &src2, error_message)) {
      return false;
    }
    *instruction =
        DecodedInstruction::Ternary(instruction_name, dst, src0, src1, src2);
    *words_consumed = 2;
    return true;
  }

  if (instruction_name == std::string_view("V_MUL_LO_U32") ||
      instruction_name == std::string_view("V_ADD_F64") ||
      instruction_name == std::string_view("V_MUL_F64") ||
      instruction_name == std::string_view("V_MUL_HI_U32") ||
      instruction_name == std::string_view("V_MUL_HI_I32") ||
      instruction_name == std::string_view("V_BCNT_U32_B32") ||
      instruction_name == std::string_view("V_BFM_B32") ||
      instruction_name == std::string_view("V_MBCNT_LO_U32_B32") ||
      instruction_name == std::string_view("V_MBCNT_HI_U32_B32") ||
      instruction_name == std::string_view("V_LSHLREV_B64") ||
      instruction_name == std::string_view("V_LSHRREV_B64") ||
      instruction_name == std::string_view("V_ASHRREV_I64")) {
    if (ExtractBits(instruction_word, 50, 9) != 0) {
      if (error_message != nullptr) {
        *error_message = "three-source VOP3 forms are not implemented";
      }
      return false;
    }
    *instruction = DecodedInstruction::Binary(instruction_name, dst, src0, src1);
    *words_consumed = 2;
    return true;
  }

  if (error_message != nullptr) {
    *error_message = "unsupported VOP3 instruction";
  }
  return false;
}

bool Gfx950BinaryDecoder::DecodeScalarDestination(
    std::uint32_t raw_value,
    InstructionOperand* operand,
    std::string* error_message) const {
  if (operand == nullptr) {
    if (error_message != nullptr) {
      *error_message = "scalar destination output must not be null";
    }
    return false;
  }
  if (raw_value >= 128) {
    if (error_message != nullptr) {
      *error_message = "unsupported scalar destination";
    }
    return false;
  }
  *operand = InstructionOperand::Sgpr(static_cast<std::uint16_t>(raw_value));
  return true;
}

bool Gfx950BinaryDecoder::DecodeVectorDestination(
    std::uint32_t raw_value,
    InstructionOperand* operand,
    std::string* error_message) const {
  if (operand == nullptr) {
    if (error_message != nullptr) {
      *error_message = "vector destination output must not be null";
    }
    return false;
  }
  if (raw_value >= 256) {
    if (error_message != nullptr) {
      *error_message = "unsupported vector destination";
    }
    return false;
  }
  *operand = InstructionOperand::Vgpr(static_cast<std::uint16_t>(raw_value));
  return true;
}

bool Gfx950BinaryDecoder::DecodeScalarSource(
    std::uint32_t raw_value,
    std::span<const std::uint32_t> literal_words,
    std::size_t* literal_words_consumed,
    InstructionOperand* operand,
    std::string* error_message) const {
  if (literal_words_consumed == nullptr || operand == nullptr) {
    if (error_message != nullptr) {
      *error_message = "scalar source outputs must not be null";
    }
    return false;
  }
  *literal_words_consumed = 0;

  if (raw_value <= 127) {
    *operand = InstructionOperand::Sgpr(static_cast<std::uint16_t>(raw_value));
    return true;
  }
  if (raw_value == 251 || raw_value == 252 || raw_value == 253) {
    *operand = InstructionOperand::Sgpr(static_cast<std::uint16_t>(raw_value));
    return true;
  }
  if (IsInlineInteger(raw_value)) {
    *operand = InstructionOperand::Imm32(DecodeInlineInteger(raw_value));
    return true;
  }
  if (raw_value == 255) {
    if (literal_words.empty()) {
      if (error_message != nullptr) {
        *error_message = "missing literal dword";
      }
      return false;
    }
    *operand = InstructionOperand::Imm32(literal_words.front());
    *literal_words_consumed = 1;
    return true;
  }

  if (error_message != nullptr) {
    *error_message = "unsupported scalar source operand encoding";
  }
  return false;
}

bool Gfx950BinaryDecoder::DecodeVectorSource(
    std::uint32_t raw_value,
    std::span<const std::uint32_t> literal_words,
    std::size_t* literal_words_consumed,
    InstructionOperand* operand,
    std::string* error_message) const {
  if (literal_words_consumed == nullptr || operand == nullptr) {
    if (error_message != nullptr) {
      *error_message = "vector source outputs must not be null";
    }
    return false;
  }
  *literal_words_consumed = 0;

  if (raw_value <= 127) {
    *operand = InstructionOperand::Sgpr(static_cast<std::uint16_t>(raw_value));
    return true;
  }
  if (raw_value == 251 || raw_value == 252 || raw_value == 253) {
    *operand = InstructionOperand::Sgpr(static_cast<std::uint16_t>(raw_value));
    return true;
  }
  if (IsInlineInteger(raw_value)) {
    *operand = InstructionOperand::Imm32(DecodeInlineInteger(raw_value));
    return true;
  }
  if (raw_value == 255) {
    if (literal_words.empty()) {
      if (error_message != nullptr) {
        *error_message = "missing literal dword";
      }
      return false;
    }
    *operand = InstructionOperand::Imm32(literal_words.front());
    *literal_words_consumed = 1;
    return true;
  }
  if (raw_value >= 256 && raw_value <= 511) {
    *operand = InstructionOperand::Vgpr(
        static_cast<std::uint16_t>(raw_value - 256));
    return true;
  }

  if (error_message != nullptr) {
    *error_message = "unsupported vector source operand encoding";
  }
  return false;
}

bool Gfx950BinaryDecoder::DecodeVectorRegisterSource(
    std::uint32_t raw_value,
    InstructionOperand* operand,
    std::string* error_message) const {
  if (operand == nullptr) {
    if (error_message != nullptr) {
      *error_message = "vector register source output must not be null";
    }
    return false;
  }
  if (raw_value >= 256) {
    if (error_message != nullptr) {
      *error_message = "unsupported VSRC1 register";
    }
    return false;
  }
  *operand = InstructionOperand::Vgpr(static_cast<std::uint16_t>(raw_value));
  return true;
}

bool Gfx950BinaryDecoder::DecodeSmemBase(std::uint32_t raw_value,
                                         InstructionOperand* operand,
                                         std::string* error_message) const {
  if (operand == nullptr) {
    if (error_message != nullptr) {
      *error_message = "smem base output must not be null";
    }
    return false;
  }
  *operand = InstructionOperand::Sgpr(
      static_cast<std::uint16_t>(raw_value << 1));
  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool Gfx950BinaryDecoder::DecodeSmemOffset(std::uint64_t instruction_word,
                                           InstructionOperand* operand,
                                           std::string* error_message) const {
  if (operand == nullptr) {
    if (error_message != nullptr) {
      *error_message = "smem offset output must not be null";
    }
    return false;
  }

  const bool use_inline_immediate = ExtractBits(instruction_word, 17, 1) != 0;
  const bool use_soffset_register = ExtractBits(instruction_word, 14, 1) != 0;

  if (use_inline_immediate) {
    *operand = InstructionOperand::Imm32(static_cast<std::uint32_t>(
        SignExtend21(static_cast<std::uint32_t>(
            ExtractBits(instruction_word, 32, 21)))));
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }

  if (use_soffset_register) {
    *operand = InstructionOperand::Sgpr(static_cast<std::uint16_t>(
        ExtractBits(instruction_word, 57, 7)));
    if (error_message != nullptr) {
      error_message->clear();
    }
    return true;
  }

  if (error_message != nullptr) {
    *error_message = "smem offset mode is not implemented";
  }
  return false;
}

bool Gfx950BinaryDecoder::DecodeFlatAddress(std::uint32_t raw_value,
                                            InstructionOperand* operand,
                                            std::string* error_message) const {
  if (operand == nullptr) {
    if (error_message != nullptr) {
      *error_message = "flat address output must not be null";
    }
    return false;
  }
  *operand = InstructionOperand::Vgpr(static_cast<std::uint16_t>(raw_value));
  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

bool Gfx950BinaryDecoder::DecodeFlatGlobalBase(
    std::uint32_t raw_value,
    InstructionOperand* operand,
    std::string* error_message) const {
  if (operand == nullptr) {
    if (error_message != nullptr) {
      *error_message = "global base output must not be null";
    }
    return false;
  }
  *operand = InstructionOperand::Sgpr(static_cast<std::uint16_t>(raw_value));
  if (error_message != nullptr) {
    error_message->clear();
  }
  return true;
}

}  // namespace mirage::sim::isa
