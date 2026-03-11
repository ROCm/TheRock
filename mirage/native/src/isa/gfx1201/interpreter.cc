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

constexpr std::uint16_t kExecPairSgprIndex = 126;
constexpr std::uint16_t kSrcVcczSgprIndex = 251;
constexpr std::uint16_t kSrcExeczSgprIndex = 252;
constexpr std::uint16_t kSrcSccSgprIndex = 253;

constexpr std::array<std::string_view, 50> kExecutableSeedOpcodes{{
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
    "V_NOT_B32",
    "V_BFREV_B32",
    "V_CVT_F32_UBYTE0",
    "V_CVT_F32_UBYTE1",
    "V_CVT_F32_UBYTE2",
    "V_CVT_F32_UBYTE3",
    "V_CVT_F32_I32",
    "V_CVT_F32_U32",
    "V_CVT_U32_F32",
    "V_CVT_I32_F32",
    "V_ADD_U32",
    "V_SUB_U32",
    "V_SUBREV_U32",
    "V_MIN_I32",
    "V_MAX_I32",
    "V_MIN_U32",
    "V_MAX_U32",
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

std::uint32_t ReverseBits32(std::uint32_t value) {
  value = ((value & 0x55555555u) << 1) | ((value >> 1) & 0x55555555u);
  value = ((value & 0x33333333u) << 2) | ((value >> 2) & 0x33333333u);
  value = ((value & 0x0f0f0f0fu) << 4) | ((value >> 4) & 0x0f0f0f0fu);
  value = ((value & 0x00ff00ffu) << 8) | ((value >> 8) & 0x00ff00ffu);
  return (value << 16) | (value >> 16);
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
  if (opcode == "V_NOT_B32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVNotB32;
    return true;
  }
  if (opcode == "V_BFREV_B32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVBfrevB32;
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
  if (opcode == "V_CVT_U32_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCvtU32F32;
    return true;
  }
  if (opcode == "V_CVT_I32_F32") {
    *compiled_opcode = Gfx1201CompiledOpcode::kVCvtI32F32;
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

std::uint32_t EvaluateVectorUnarySeedInstruction(std::string_view opcode,
                                                 std::uint32_t value) {
  if (opcode == "V_NOT_B32") {
    return ~value;
  }
  if (opcode == "V_BFREV_B32") {
    return ReverseBits32(value);
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
  if (opcode == "V_CVT_U32_F32") {
    return TruncateFloatToU32(BitCast<float>(value));
  }
  if (opcode == "V_CVT_I32_F32") {
    return BitCast<std::uint32_t>(TruncateFloatToI32(BitCast<float>(value)));
  }
  return value;
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
      instruction.opcode == "V_CVT_F32_UBYTE0" ||
      instruction.opcode == "V_CVT_F32_UBYTE1" ||
      instruction.opcode == "V_CVT_F32_UBYTE2" ||
      instruction.opcode == "V_CVT_F32_UBYTE3" ||
      instruction.opcode == "V_CVT_F32_I32" ||
      instruction.opcode == "V_CVT_F32_U32" || instruction.opcode == "V_CVT_U32_F32" ||
      instruction.opcode == "V_CVT_I32_F32") {
    if (!ValidateOperandCount(instruction, 2, error_message)) {
      return false;
    }
    for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
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

  if (instruction.opcode == "V_ADD_U32" || instruction.opcode == "V_SUB_U32" ||
      instruction.opcode == "V_SUBREV_U32" || instruction.opcode == "V_MIN_I32" ||
      instruction.opcode == "V_MAX_I32" || instruction.opcode == "V_MIN_U32" ||
      instruction.opcode == "V_MAX_U32" ||
      instruction.opcode == "V_LSHRREV_B32" ||
      instruction.opcode == "V_ASHRREV_I32" ||
      instruction.opcode == "V_LSHLREV_B32" || instruction.opcode == "V_AND_B32" ||
      instruction.opcode == "V_OR_B32" || instruction.opcode == "V_XOR_B32") {
    if (!ValidateOperandCount(instruction, 3, error_message)) {
      return false;
    }
    for (std::size_t lane_index = 0; lane_index < WaveExecutionState::kLaneCount;
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
          EvaluateVectorBinarySeedInstruction(instruction.opcode, lhs, rhs);
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
    case Gfx1201CompiledOpcode::kVNotB32:
    case Gfx1201CompiledOpcode::kVBfrevB32:
    case Gfx1201CompiledOpcode::kVCvtF32Ubyte0:
    case Gfx1201CompiledOpcode::kVCvtF32Ubyte1:
    case Gfx1201CompiledOpcode::kVCvtF32Ubyte2:
    case Gfx1201CompiledOpcode::kVCvtF32Ubyte3:
    case Gfx1201CompiledOpcode::kVCvtF32I32:
    case Gfx1201CompiledOpcode::kVCvtF32U32:
    case Gfx1201CompiledOpcode::kVCvtU32F32:
    case Gfx1201CompiledOpcode::kVCvtI32F32:
    case Gfx1201CompiledOpcode::kVAddU32:
    case Gfx1201CompiledOpcode::kVSubU32:
    case Gfx1201CompiledOpcode::kVSubrevU32:
    case Gfx1201CompiledOpcode::kVMinI32:
    case Gfx1201CompiledOpcode::kVMaxI32:
    case Gfx1201CompiledOpcode::kVMinU32:
    case Gfx1201CompiledOpcode::kVMaxU32:
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
