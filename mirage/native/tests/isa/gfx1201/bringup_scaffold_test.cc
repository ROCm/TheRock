#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/common/wave_execution_state.h"
#include "lib/sim/isa/gfx1201/architecture_profile.h"
#include "lib/sim/isa/gfx1201/binary_decoder.h"
#include "lib/sim/isa/gfx1201/interpreter.h"

namespace {

bool Expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << message << '\n';
    return false;
  }
  return true;
}

constexpr std::uint32_t SetBits(std::uint32_t word,
                                std::uint32_t value,
                                std::uint32_t bit_offset,
                                std::uint32_t bit_count) {
  const std::uint32_t mask =
      (bit_count == 32) ? 0xffffffffu : ((1u << bit_count) - 1u);
  return word | ((value & mask) << bit_offset);
}

constexpr std::uint32_t MakeSopp(std::uint32_t op, std::uint32_t simm16 = 0) {
  std::uint32_t word = 0;
  word = SetBits(word, 0x17f, 23, 9);
  word = SetBits(word, op, 16, 7);
  word = SetBits(word, simm16, 0, 16);
  return word;
}

constexpr std::uint32_t MakeSopk(std::uint32_t op,
                                 std::uint32_t sdst,
                                 std::uint32_t simm16) {
  std::uint32_t word = 0;
  word = SetBits(word, 0xb, 28, 4);
  word = SetBits(word, op, 23, 5);
  word = SetBits(word, sdst, 16, 7);
  word = SetBits(word, simm16, 0, 16);
  return word;
}

}  // namespace

int main() {
  using namespace mirage::sim::isa;

  const InstructionCatalogMetadata& metadata =
      GetGfx1201ImportedInstructionMetadata();
  if (!Expect(metadata.gfx_target == "gfx1201", "expected gfx1201 target") ||
      !Expect(metadata.architecture_name == "AMD RDNA 4",
              "expected AMD RDNA 4 architecture") ||
      !Expect(metadata.source_xml == "amdgpu_isa_rdna4.xml",
              "expected RDNA4 source xml") ||
      !Expect(metadata.instruction_count == 1264u,
              "expected imported instruction count") ||
      !Expect(metadata.encoding_count == 5062u,
              "expected imported encoding count")) {
    return 1;
  }

  const auto support_buckets = GetGfx1201SupportBucketSummaries();
  if (!Expect(support_buckets.size() == 5u,
              "expected five support buckets") ||
      !Expect(support_buckets.front().instruction_count == 363u,
              "expected transferable_full count") ||
      !Expect(support_buckets.back().instruction_count == 668u,
              "expected new_vs_gfx950 count")) {
    return 1;
  }

  Gfx1201BinaryDecoder decoder;
  if (!Expect(decoder.Phase0EncodingFocus().size() == 12u,
              "expected phase-0 encoding focus list") ||
      !Expect(decoder.Phase1EncodingFocus().size() == 8u,
              "expected phase-1 encoding focus list") ||
      !Expect(decoder.Phase0ComputeSeeds().size() == 12u,
              "expected phase-0 compute seed list") ||
      !Expect(decoder.Phase0ComputeSelectorRules().size() == 12u,
              "expected phase-0 selector rule list") ||
      !Expect(decoder.Phase0ExecutableOpcodes().size() == 325u,
              "expected phase-0 executable opcode slice") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_ADD_U32"),
              "expected S_ADD_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_SUB_U32"),
              "expected S_SUB_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_CMP_EQ_I32"),
              "expected S_CMP_EQ_I32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_CMP_EQ_U32"),
              "expected S_CMP_EQ_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_CMP_GE_I32"),
              "expected S_CMP_GE_I32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_CMP_LT_U32"),
              "expected S_CMP_LT_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_CBRANCH_SCC1"),
              "expected S_CBRANCH_SCC1 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_CBRANCH_VCCNZ"),
              "expected S_CBRANCH_VCCNZ executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_CBRANCH_EXECZ"),
              "expected S_CBRANCH_EXECZ executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_NOT_B32"),
              "expected V_NOT_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CLZ_I32_U32"),
              "expected V_CLZ_I32_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CLS_I32"),
              "expected V_CLS_I32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMP_EQ_U32"),
              "expected V_CMP_EQ_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMPX_EQ_U32"),
              "expected V_CMPX_EQ_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMP_EQ_I16"),
              "expected V_CMP_EQ_I16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMPX_EQ_U16"),
              "expected V_CMPX_EQ_U16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMP_EQ_F16"),
              "expected V_CMP_EQ_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMPX_CLASS_F16"),
              "expected V_CMPX_CLASS_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMP_EQ_F32"),
              "expected V_CMP_EQ_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMP_O_F32"),
              "expected V_CMP_O_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMPX_CLASS_F32"),
              "expected V_CMPX_CLASS_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMPX_U_F32"),
              "expected V_CMPX_U_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMP_EQ_F64"),
              "expected V_CMP_EQ_F64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMPX_CLASS_F64"),
              "expected V_CMPX_CLASS_F64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMP_EQ_I64"),
              "expected V_CMP_EQ_I64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMPX_EQ_U64"),
              "expected V_CMPX_EQ_U64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMP_GE_I32"),
              "expected V_CMP_GE_I32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CMPX_GE_I32"),
              "expected V_CMPX_GE_I32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_F32_UBYTE3"),
              "expected V_CVT_F32_UBYTE3 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_READFIRSTLANE_B32"),
              "expected V_READFIRSTLANE_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MOVRELD_B32"),
              "expected V_MOVRELD_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MOVRELS_B32"),
              "expected V_MOVRELS_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MOVRELSD_B32"),
              "expected V_MOVRELSD_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MOVRELSD_2_B32"),
              "expected V_MOVRELSD_2_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SWAPREL_B32"),
              "expected V_SWAPREL_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_ADD_U32"),
              "expected V_ADD_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SUB_U32"),
              "expected V_SUB_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SUBREV_U32"),
              "expected V_SUBREV_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CNDMASK_B32"),
              "expected V_CNDMASK_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_XOR_B32"),
              "expected V_XOR_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_NOP"),
              "expected V_NOP executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_PIPEFLUSH"),
              "expected V_PIPEFLUSH executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MOV_B16"),
              "expected V_MOV_B16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_PERMLANE64_B32"),
              "expected V_PERMLANE64_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SWAP_B32"),
              "expected V_SWAP_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SWAP_B16"),
              "expected V_SWAP_B16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_NOT_B16"),
              "expected V_NOT_B16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_F32_I32"),
              "expected V_CVT_F32_I32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_F32_FP8"),
              "expected V_CVT_F32_FP8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_F32_BF8"),
              "expected V_CVT_F32_BF8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_PK_F32_FP8"),
              "expected V_CVT_PK_F32_FP8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_PK_F32_BF8"),
              "expected V_CVT_PK_F32_BF8 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_NEAREST_I32_F32"),
              "expected V_CVT_NEAREST_I32_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_F16_F32"),
              "expected V_CVT_F16_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_F16_I16"),
              "expected V_CVT_F16_I16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_F16_U16"),
              "expected V_CVT_F16_U16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_F32_F16"),
              "expected V_CVT_F32_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_I16_F16"),
              "expected V_CVT_I16_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_U16_F16"),
              "expected V_CVT_U16_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SAT_PK_U8_I16"),
              "expected V_SAT_PK_U8_I16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_OFF_F32_I4"),
              "expected V_CVT_OFF_F32_I4 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_NORM_I16_F16"),
              "expected V_CVT_NORM_I16_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_NORM_U16_F16"),
              "expected V_CVT_NORM_U16_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_RCP_F16"),
              "expected V_RCP_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_RSQ_F16"),
              "expected V_RSQ_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SQRT_F16"),
              "expected V_SQRT_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_EXP_F16"),
              "expected V_EXP_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_LOG_F16"),
              "expected V_LOG_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SIN_F16"),
              "expected V_SIN_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_COS_F16"),
              "expected V_COS_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FREXP_EXP_I16_F16"),
              "expected V_FREXP_EXP_I16_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FREXP_MANT_F16"),
              "expected V_FREXP_MANT_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FRACT_F16"),
              "expected V_FRACT_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_TRUNC_F16"),
              "expected V_TRUNC_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CEIL_F16"),
              "expected V_CEIL_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_RNDNE_F16"),
              "expected V_RNDNE_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FLOOR_F16"),
              "expected V_FLOOR_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_ADD_F16"),
              "expected V_ADD_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SUB_F16"),
              "expected V_SUB_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SUBREV_F16"),
              "expected V_SUBREV_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MUL_F16"),
              "expected V_MUL_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_PK_RTZ_F16_F32"),
              "expected V_CVT_PK_RTZ_F16_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_LDEXP_F16"),
              "expected V_LDEXP_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MIN_NUM_F16"),
              "expected V_MIN_NUM_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MAX_NUM_F16"),
              "expected V_MAX_NUM_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_ADD_F32"),
              "expected V_ADD_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SUB_F32"),
              "expected V_SUB_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SUBREV_F32"),
              "expected V_SUBREV_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MUL_F32"),
              "expected V_MUL_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MIN_NUM_F32"),
              "expected V_MIN_NUM_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MAX_NUM_F32"),
              "expected V_MAX_NUM_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_ADD_F64"),
              "expected V_ADD_F64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MUL_F64"),
              "expected V_MUL_F64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MIN_NUM_F64"),
              "expected V_MIN_NUM_F64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MAX_NUM_F64"),
              "expected V_MAX_NUM_F64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_XNOR_B32"),
              "expected V_XNOR_B32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MUL_I32_I24"),
              "expected V_MUL_I32_I24 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MUL_HI_I32_I24"),
              "expected V_MUL_HI_I32_I24 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MUL_U32_U24"),
              "expected V_MUL_U32_U24 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MUL_HI_U32_U24"),
              "expected V_MUL_HI_U32_U24 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_LSHLREV_B64"),
              "expected V_LSHLREV_B64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_ADD_CO_CI_U32"),
              "expected V_ADD_CO_CI_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SUB_CO_CI_U32"),
              "expected V_SUB_CO_CI_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SUBREV_CO_CI_U32"),
              "expected V_SUBREV_CO_CI_U32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_MUL_DX9_ZERO_F32"),
              "expected V_MUL_DX9_ZERO_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FMAC_F32"),
              "expected V_FMAC_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FMAC_F16"),
              "expected V_FMAC_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FMAMK_F16"),
              "expected V_FMAMK_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FMAAK_F16"),
              "expected V_FMAAK_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_PK_FMAC_F16"),
              "expected V_PK_FMAC_F16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_I32_I16"),
              "expected V_CVT_I32_I16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_U32_U16"),
              "expected V_CVT_U32_U16 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_F64_F32"),
              "expected V_CVT_F64_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_I32_F64"),
              "expected V_CVT_I32_F64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_RCP_F32"),
              "expected V_RCP_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_LOG_F32"),
              "expected V_LOG_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_SQRT_F64"),
              "expected V_SQRT_F64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FREXP_EXP_I32_F32"),
              "expected V_FREXP_EXP_I32_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FRACT_F64"),
              "expected V_FRACT_F64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_TRUNC_F32"),
              "expected V_TRUNC_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_FLOOR_F64"),
              "expected V_FLOOR_F64 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("V_CVT_I32_F32"),
              "expected V_CVT_I32_F32 executable decode support") ||
      !Expect(decoder.SupportsPhase0ExecutableOpcode("S_MOV_B32"),
              "expected S_MOV_B32 executable decode support") ||
      !Expect(!decoder.SupportsPhase0ExecutableOpcode("V_DOT2_F32_F16"),
              "expected V_DOT2_F32_F16 to remain outside executable decode slice")) {
    return 1;
  }

  DecodedInstruction decoded_instruction;
  std::size_t words_consumed = 99;
  std::string error_message;
  const std::array<std::uint32_t, 1> route_only_words{0u};
  if (!Expect(!decoder.DecodeInstruction(route_only_words, &decoded_instruction,
                                         &words_consumed, &error_message),
              "decoder should still fail outside executable seed slice") ||
      !Expect(words_consumed == 0u, "expected no words consumed on route-only miss") ||
      !Expect(error_message.find("ENC_VOP2 opcode 0") != std::string::npos,
              "expected phase-0 route in error") ||
      !Expect(error_message.find("no matching seed entry") != std::string::npos,
              "expected seed-aware route miss in error")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> endpgm_words{MakeSopp(48u)};
  if (!Expect(decoder.DecodeInstruction(endpgm_words, &decoded_instruction,
                                        &words_consumed, &error_message),
              "expected S_ENDPGM decode success") ||
      !Expect(words_consumed == 1u, "expected one dword consumed") ||
      !Expect(decoded_instruction.opcode == "S_ENDPGM",
              "expected S_ENDPGM opcode") ||
      !Expect(decoded_instruction.operand_count == 0u,
              "expected S_ENDPGM nullary decode")) {
    return 1;
  }

  Gfx1201Interpreter interpreter;
  if (!Expect(interpreter.ExecutableSeedOpcodes().size() == 325u,
              "expected executable seed opcode list") ||
      !Expect(interpreter.Supports("S_ENDPGM"),
              "expected interpreter support for S_ENDPGM") ||
      !Expect(interpreter.Supports("S_ADD_U32"),
              "expected interpreter support for S_ADD_U32") ||
      !Expect(interpreter.Supports("S_ADD_I32"),
              "expected interpreter support for S_ADD_I32") ||
      !Expect(interpreter.Supports("S_SUB_U32"),
              "expected interpreter support for S_SUB_U32") ||
      !Expect(interpreter.Supports("S_CMP_EQ_U32"),
              "expected interpreter support for S_CMP_EQ_U32") ||
      !Expect(interpreter.Supports("S_CMP_LG_U32"),
              "expected interpreter support for S_CMP_LG_U32") ||
      !Expect(interpreter.Supports("S_CMP_EQ_I32"),
              "expected interpreter support for S_CMP_EQ_I32") ||
      !Expect(interpreter.Supports("S_CMP_GT_I32"),
              "expected interpreter support for S_CMP_GT_I32") ||
      !Expect(interpreter.Supports("S_CMP_LE_U32"),
              "expected interpreter support for S_CMP_LE_U32") ||
      !Expect(interpreter.Supports("S_CMP_GE_I32"),
              "expected interpreter support for S_CMP_GE_I32") ||
      !Expect(interpreter.Supports("S_CMP_LT_I32"),
              "expected interpreter support for S_CMP_LT_I32") ||
      !Expect(interpreter.Supports("S_CMP_GE_U32"),
              "expected interpreter support for S_CMP_GE_U32") ||
      !Expect(interpreter.Supports("S_CMP_LT_U32"),
              "expected interpreter support for S_CMP_LT_U32") ||
      !Expect(interpreter.Supports("S_BRANCH"),
              "expected interpreter support for S_BRANCH") ||
      !Expect(interpreter.Supports("S_CBRANCH_SCC0"),
              "expected interpreter support for S_CBRANCH_SCC0") ||
      !Expect(interpreter.Supports("S_CBRANCH_SCC1"),
              "expected interpreter support for S_CBRANCH_SCC1") ||
      !Expect(interpreter.Supports("S_CBRANCH_VCCZ"),
              "expected interpreter support for S_CBRANCH_VCCZ") ||
      !Expect(interpreter.Supports("S_CBRANCH_VCCNZ"),
              "expected interpreter support for S_CBRANCH_VCCNZ") ||
      !Expect(interpreter.Supports("S_CBRANCH_EXECZ"),
              "expected interpreter support for S_CBRANCH_EXECZ") ||
      !Expect(interpreter.Supports("S_CBRANCH_EXECNZ"),
              "expected interpreter support for S_CBRANCH_EXECNZ") ||
      !Expect(interpreter.Supports("S_MOV_B32"),
              "expected interpreter support for S_MOV_B32") ||
      !Expect(interpreter.Supports("V_NOT_B32"),
              "expected interpreter support for V_NOT_B32") ||
      !Expect(interpreter.Supports("V_CLZ_I32_U32"),
              "expected interpreter support for V_CLZ_I32_U32") ||
      !Expect(interpreter.Supports("V_CLS_I32"),
              "expected interpreter support for V_CLS_I32") ||
      !Expect(interpreter.Supports("V_CMP_EQ_U32"),
              "expected interpreter support for V_CMP_EQ_U32") ||
      !Expect(interpreter.Supports("V_CMPX_EQ_U32"),
              "expected interpreter support for V_CMPX_EQ_U32") ||
      !Expect(interpreter.Supports("V_CMP_EQ_I16"),
              "expected interpreter support for V_CMP_EQ_I16") ||
      !Expect(interpreter.Supports("V_CMPX_EQ_U16"),
              "expected interpreter support for V_CMPX_EQ_U16") ||
      !Expect(interpreter.Supports("V_CMP_EQ_F16"),
              "expected interpreter support for V_CMP_EQ_F16") ||
      !Expect(interpreter.Supports("V_CMPX_CLASS_F16"),
              "expected interpreter support for V_CMPX_CLASS_F16") ||
      !Expect(interpreter.Supports("V_CMP_EQ_F32"),
              "expected interpreter support for V_CMP_EQ_F32") ||
      !Expect(interpreter.Supports("V_CMP_O_F32"),
              "expected interpreter support for V_CMP_O_F32") ||
      !Expect(interpreter.Supports("V_CMPX_CLASS_F32"),
              "expected interpreter support for V_CMPX_CLASS_F32") ||
      !Expect(interpreter.Supports("V_CMPX_U_F32"),
              "expected interpreter support for V_CMPX_U_F32") ||
      !Expect(interpreter.Supports("V_CMP_EQ_F64"),
              "expected interpreter support for V_CMP_EQ_F64") ||
      !Expect(interpreter.Supports("V_CMPX_CLASS_F64"),
              "expected interpreter support for V_CMPX_CLASS_F64") ||
      !Expect(interpreter.Supports("V_CMP_EQ_I64"),
              "expected interpreter support for V_CMP_EQ_I64") ||
      !Expect(interpreter.Supports("V_CMPX_EQ_U64"),
              "expected interpreter support for V_CMPX_EQ_U64") ||
      !Expect(interpreter.Supports("V_CMP_GE_I32"),
              "expected interpreter support for V_CMP_GE_I32") ||
      !Expect(interpreter.Supports("V_CMPX_GE_I32"),
              "expected interpreter support for V_CMPX_GE_I32") ||
      !Expect(interpreter.Supports("V_NOP"),
              "expected interpreter support for V_NOP") ||
      !Expect(interpreter.Supports("V_PIPEFLUSH"),
              "expected interpreter support for V_PIPEFLUSH") ||
      !Expect(interpreter.Supports("V_MOV_B16"),
              "expected interpreter support for V_MOV_B16") ||
      !Expect(interpreter.Supports("V_PERMLANE64_B32"),
              "expected interpreter support for V_PERMLANE64_B32") ||
      !Expect(interpreter.Supports("V_SWAP_B32"),
              "expected interpreter support for V_SWAP_B32") ||
      !Expect(interpreter.Supports("V_SWAP_B16"),
              "expected interpreter support for V_SWAP_B16") ||
      !Expect(interpreter.Supports("V_NOT_B16"),
              "expected interpreter support for V_NOT_B16") ||
      !Expect(interpreter.Supports("V_BFREV_B32"),
              "expected interpreter support for V_BFREV_B32") ||
      !Expect(interpreter.Supports("V_CVT_F32_UBYTE0"),
              "expected interpreter support for V_CVT_F32_UBYTE0") ||
      !Expect(interpreter.Supports("V_CVT_F32_UBYTE3"),
              "expected interpreter support for V_CVT_F32_UBYTE3") ||
      !Expect(interpreter.Supports("V_CVT_F32_I32"),
              "expected interpreter support for V_CVT_F32_I32") ||
      !Expect(interpreter.Supports("V_CVT_F32_FP8"),
              "expected interpreter support for V_CVT_F32_FP8") ||
      !Expect(interpreter.Supports("V_CVT_F32_BF8"),
              "expected interpreter support for V_CVT_F32_BF8") ||
      !Expect(interpreter.Supports("V_CVT_PK_F32_FP8"),
              "expected interpreter support for V_CVT_PK_F32_FP8") ||
      !Expect(interpreter.Supports("V_CVT_PK_F32_BF8"),
              "expected interpreter support for V_CVT_PK_F32_BF8") ||
      !Expect(interpreter.Supports("V_CVT_NEAREST_I32_F32"),
              "expected interpreter support for V_CVT_NEAREST_I32_F32") ||
      !Expect(interpreter.Supports("V_CVT_F16_F32"),
              "expected interpreter support for V_CVT_F16_F32") ||
      !Expect(interpreter.Supports("V_CVT_F16_I16"),
              "expected interpreter support for V_CVT_F16_I16") ||
      !Expect(interpreter.Supports("V_CVT_F16_U16"),
              "expected interpreter support for V_CVT_F16_U16") ||
      !Expect(interpreter.Supports("V_CVT_F32_F16"),
              "expected interpreter support for V_CVT_F32_F16") ||
      !Expect(interpreter.Supports("V_CVT_I16_F16"),
              "expected interpreter support for V_CVT_I16_F16") ||
      !Expect(interpreter.Supports("V_CVT_U16_F16"),
              "expected interpreter support for V_CVT_U16_F16") ||
      !Expect(interpreter.Supports("V_SAT_PK_U8_I16"),
              "expected interpreter support for V_SAT_PK_U8_I16") ||
      !Expect(interpreter.Supports("V_CVT_OFF_F32_I4"),
              "expected interpreter support for V_CVT_OFF_F32_I4") ||
      !Expect(interpreter.Supports("V_CVT_NORM_I16_F16"),
              "expected interpreter support for V_CVT_NORM_I16_F16") ||
      !Expect(interpreter.Supports("V_CVT_NORM_U16_F16"),
              "expected interpreter support for V_CVT_NORM_U16_F16") ||
      !Expect(interpreter.Supports("V_RCP_F16"),
              "expected interpreter support for V_RCP_F16") ||
      !Expect(interpreter.Supports("V_RSQ_F16"),
              "expected interpreter support for V_RSQ_F16") ||
      !Expect(interpreter.Supports("V_SQRT_F16"),
              "expected interpreter support for V_SQRT_F16") ||
      !Expect(interpreter.Supports("V_EXP_F16"),
              "expected interpreter support for V_EXP_F16") ||
      !Expect(interpreter.Supports("V_LOG_F16"),
              "expected interpreter support for V_LOG_F16") ||
      !Expect(interpreter.Supports("V_SIN_F16"),
              "expected interpreter support for V_SIN_F16") ||
      !Expect(interpreter.Supports("V_COS_F16"),
              "expected interpreter support for V_COS_F16") ||
      !Expect(interpreter.Supports("V_FREXP_EXP_I16_F16"),
              "expected interpreter support for V_FREXP_EXP_I16_F16") ||
      !Expect(interpreter.Supports("V_FREXP_MANT_F16"),
              "expected interpreter support for V_FREXP_MANT_F16") ||
      !Expect(interpreter.Supports("V_FRACT_F16"),
              "expected interpreter support for V_FRACT_F16") ||
      !Expect(interpreter.Supports("V_TRUNC_F16"),
              "expected interpreter support for V_TRUNC_F16") ||
      !Expect(interpreter.Supports("V_CEIL_F16"),
              "expected interpreter support for V_CEIL_F16") ||
      !Expect(interpreter.Supports("V_RNDNE_F16"),
              "expected interpreter support for V_RNDNE_F16") ||
      !Expect(interpreter.Supports("V_FLOOR_F16"),
              "expected interpreter support for V_FLOOR_F16") ||
      !Expect(interpreter.Supports("V_ADD_F16"),
              "expected interpreter support for V_ADD_F16") ||
      !Expect(interpreter.Supports("V_SUB_F16"),
              "expected interpreter support for V_SUB_F16") ||
      !Expect(interpreter.Supports("V_SUBREV_F16"),
              "expected interpreter support for V_SUBREV_F16") ||
      !Expect(interpreter.Supports("V_MUL_F16"),
              "expected interpreter support for V_MUL_F16") ||
      !Expect(interpreter.Supports("V_CVT_PK_RTZ_F16_F32"),
              "expected interpreter support for V_CVT_PK_RTZ_F16_F32") ||
      !Expect(interpreter.Supports("V_LDEXP_F16"),
              "expected interpreter support for V_LDEXP_F16") ||
      !Expect(interpreter.Supports("V_MIN_NUM_F16"),
              "expected interpreter support for V_MIN_NUM_F16") ||
      !Expect(interpreter.Supports("V_MAX_NUM_F16"),
              "expected interpreter support for V_MAX_NUM_F16") ||
      !Expect(interpreter.Supports("V_ADD_F32"),
              "expected interpreter support for V_ADD_F32") ||
      !Expect(interpreter.Supports("V_SUB_F32"),
              "expected interpreter support for V_SUB_F32") ||
      !Expect(interpreter.Supports("V_SUBREV_F32"),
              "expected interpreter support for V_SUBREV_F32") ||
      !Expect(interpreter.Supports("V_MUL_F32"),
              "expected interpreter support for V_MUL_F32") ||
      !Expect(interpreter.Supports("V_MIN_NUM_F32"),
              "expected interpreter support for V_MIN_NUM_F32") ||
      !Expect(interpreter.Supports("V_MAX_NUM_F32"),
              "expected interpreter support for V_MAX_NUM_F32") ||
      !Expect(interpreter.Supports("V_ADD_F64"),
              "expected interpreter support for V_ADD_F64") ||
      !Expect(interpreter.Supports("V_MUL_F64"),
              "expected interpreter support for V_MUL_F64") ||
      !Expect(interpreter.Supports("V_MIN_NUM_F64"),
              "expected interpreter support for V_MIN_NUM_F64") ||
      !Expect(interpreter.Supports("V_MAX_NUM_F64"),
              "expected interpreter support for V_MAX_NUM_F64") ||
      !Expect(interpreter.Supports("V_XNOR_B32"),
              "expected interpreter support for V_XNOR_B32") ||
      !Expect(interpreter.Supports("V_MUL_I32_I24"),
              "expected interpreter support for V_MUL_I32_I24") ||
      !Expect(interpreter.Supports("V_MUL_HI_I32_I24"),
              "expected interpreter support for V_MUL_HI_I32_I24") ||
      !Expect(interpreter.Supports("V_MUL_U32_U24"),
              "expected interpreter support for V_MUL_U32_U24") ||
      !Expect(interpreter.Supports("V_MUL_HI_U32_U24"),
              "expected interpreter support for V_MUL_HI_U32_U24") ||
      !Expect(interpreter.Supports("V_LSHLREV_B64"),
              "expected interpreter support for V_LSHLREV_B64") ||
      !Expect(interpreter.Supports("V_CVT_I32_I16"),
              "expected interpreter support for V_CVT_I32_I16") ||
      !Expect(interpreter.Supports("V_CVT_U32_U16"),
              "expected interpreter support for V_CVT_U32_U16") ||
      !Expect(interpreter.Supports("V_CVT_F32_U32"),
              "expected interpreter support for V_CVT_F32_U32") ||
      !Expect(interpreter.Supports("V_CVT_F64_F32"),
              "expected interpreter support for V_CVT_F64_F32") ||
      !Expect(interpreter.Supports("V_CVT_I32_F64"),
              "expected interpreter support for V_CVT_I32_F64") ||
      !Expect(interpreter.Supports("V_RCP_F32"),
              "expected interpreter support for V_RCP_F32") ||
      !Expect(interpreter.Supports("V_LOG_F32"),
              "expected interpreter support for V_LOG_F32") ||
      !Expect(interpreter.Supports("V_SQRT_F64"),
              "expected interpreter support for V_SQRT_F64") ||
      !Expect(interpreter.Supports("V_FREXP_EXP_I32_F32"),
              "expected interpreter support for V_FREXP_EXP_I32_F32") ||
      !Expect(interpreter.Supports("V_FRACT_F64"),
              "expected interpreter support for V_FRACT_F64") ||
      !Expect(interpreter.Supports("V_TRUNC_F32"),
              "expected interpreter support for V_TRUNC_F32") ||
      !Expect(interpreter.Supports("V_FLOOR_F64"),
              "expected interpreter support for V_FLOOR_F64") ||
      !Expect(interpreter.Supports("V_CVT_U32_F32"),
              "expected interpreter support for V_CVT_U32_F32") ||
      !Expect(interpreter.Supports("V_CVT_I32_F32"),
              "expected interpreter support for V_CVT_I32_F32") ||
      !Expect(interpreter.Supports("V_READFIRSTLANE_B32"),
              "expected interpreter support for V_READFIRSTLANE_B32") ||
      !Expect(interpreter.Supports("V_MOVRELD_B32"),
              "expected interpreter support for V_MOVRELD_B32") ||
      !Expect(interpreter.Supports("V_MOVRELS_B32"),
              "expected interpreter support for V_MOVRELS_B32") ||
      !Expect(interpreter.Supports("V_MOVRELSD_B32"),
              "expected interpreter support for V_MOVRELSD_B32") ||
      !Expect(interpreter.Supports("V_MOVRELSD_2_B32"),
              "expected interpreter support for V_MOVRELSD_2_B32") ||
      !Expect(interpreter.Supports("V_SWAPREL_B32"),
              "expected interpreter support for V_SWAPREL_B32") ||
      !Expect(interpreter.Supports("V_ADD_U32"),
              "expected interpreter support for V_ADD_U32") ||
      !Expect(interpreter.Supports("V_SUB_U32"),
              "expected interpreter support for V_SUB_U32") ||
      !Expect(interpreter.Supports("V_SUBREV_U32"),
              "expected interpreter support for V_SUBREV_U32") ||
      !Expect(interpreter.Supports("V_CNDMASK_B32"),
              "expected interpreter support for V_CNDMASK_B32") ||
      !Expect(interpreter.Supports("V_MIN_I32"),
              "expected interpreter support for V_MIN_I32") ||
      !Expect(interpreter.Supports("V_LSHRREV_B32"),
              "expected interpreter support for V_LSHRREV_B32") ||
      !Expect(interpreter.Supports("V_XOR_B32"),
              "expected interpreter support for V_XOR_B32") ||
      !Expect(interpreter.Supports("V_MOV_B32"),
              "expected interpreter support for V_MOV_B32") ||
      !Expect(interpreter.Supports("V_MUL_DX9_ZERO_F32"),
              "expected interpreter support for V_MUL_DX9_ZERO_F32") ||
      !Expect(interpreter.Supports("V_FMAC_F32"),
              "expected interpreter support for V_FMAC_F32") ||
      !Expect(interpreter.Supports("V_FMAC_F16"),
              "expected interpreter support for V_FMAC_F16") ||
      !Expect(interpreter.Supports("V_FMAMK_F16"),
              "expected interpreter support for V_FMAMK_F16") ||
      !Expect(interpreter.Supports("V_FMAAK_F16"),
              "expected interpreter support for V_FMAAK_F16") ||
      !Expect(interpreter.Supports("V_PK_FMAC_F16"),
              "expected interpreter support for V_PK_FMAC_F16") ||
      !Expect(interpreter.Supports("V_ADD_CO_CI_U32"),
              "expected interpreter support for V_ADD_CO_CI_U32") ||
      !Expect(interpreter.Supports("V_SUB_CO_CI_U32"),
              "expected interpreter support for V_SUB_CO_CI_U32") ||
      !Expect(interpreter.Supports("V_SUBREV_CO_CI_U32"),
              "expected interpreter support for V_SUBREV_CO_CI_U32") ||
      !Expect(!interpreter.Supports("V_DOT2_F32_F16"),
              "expected interpreter to reject unsupported seed opcode") ||
      !Expect(interpreter.CarryOverFamilyFocus().size() == 7u,
              "expected carry-over family focus list") ||
      !Expect(interpreter.Rdna4DeltaFamilyFocus().size() == 10u,
              "expected RDNA4 delta family focus list")) {
    return 1;
  }

  const std::array<DecodedInstruction, 3> supported_program{
      DecodedInstruction::Unary("S_MOVK_I32", InstructionOperand::Sgpr(1),
                                InstructionOperand::Imm32(7u)),
      DecodedInstruction::Binary("S_ADD_U32", InstructionOperand::Sgpr(2),
                                 InstructionOperand::Sgpr(1),
                                 InstructionOperand::Imm32(5u)),
      DecodedInstruction::Nullary("S_ENDPGM"),
  };
  std::vector<Gfx1201CompiledInstruction> compiled_program;
  if (!Expect(interpreter.CompileProgram(supported_program, &compiled_program,
                                         &error_message),
              "expected compile success for executable seed slice") ||
      !Expect(compiled_program.size() == supported_program.size(),
              "expected compiled instruction count")) {
    return 1;
  }

  WaveExecutionState state;
  if (!Expect(interpreter.ExecuteProgram(supported_program, &state, &error_message),
              "expected decoded execution success for executable seed slice") ||
      !Expect(state.lane_count == 32u,
              "expected gfx1201 execution to normalize to wave32") ||
      !Expect(state.exec_mask == 0xffffffffULL,
              "expected gfx1201 execution to clamp exec to wave32") ||
      !Expect(state.sgprs[1] == 7u, "expected decoded execution to write SGPR") ||
      !Expect(state.sgprs[2] == 12u, "expected decoded execution to add into SGPR") ||
      !Expect(state.halted, "expected decoded execution to halt")) {
    return 1;
  }

  const std::array<DecodedInstruction, 1> unsupported_program{
      DecodedInstruction::Binary("V_DOT2_F32_F16", InstructionOperand::Vgpr(0),
                                 InstructionOperand::Vgpr(1),
                                 InstructionOperand::Vgpr(2)),
  };
  if (!Expect(!interpreter.CompileProgram(unsupported_program, &compiled_program,
                                          &error_message),
              "expected unsupported program compile failure") ||
      !Expect(error_message.find("V_DOT2_F32_F16 is not in the executable seed slice") !=
                  std::string::npos,
              "expected unsupported opcode message") ||
      !Expect(error_message.find("RDNA4 delta families") != std::string::npos,
              "expected remaining bring-up summary in error")) {
    return 1;
  }

  return 0;
}
