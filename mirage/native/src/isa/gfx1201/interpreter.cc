#include "lib/sim/isa/gfx1201/interpreter.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "lib/sim/isa/gfx1201/support_catalog.h"

namespace mirage::sim::isa {
namespace {

constexpr std::uint16_t kExecPairSgprIndex = 126;
constexpr std::uint16_t kSrcVcczSgprIndex = 251;
constexpr std::uint16_t kSrcExeczSgprIndex = 252;
constexpr std::uint16_t kSrcSccSgprIndex = 253;

constexpr std::array<std::string_view, 5> kExecutableSeedOpcodes{{
    "S_ENDPGM",
    "S_NOP",
    "S_MOV_B32",
    "S_MOVK_I32",
    "V_MOV_B32",
}};

bool IsExecutableSeedOpcode(std::string_view opcode) {
  for (std::string_view supported_opcode : kExecutableSeedOpcodes) {
    if (supported_opcode == opcode) {
      return true;
    }
  }
  return false;
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
  *compiled_opcode = Gfx1201CompiledOpcode::kUnknown;
  return false;
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

bool ExecuteDecodedSeedInstruction(const DecodedInstruction& instruction,
                                   WaveExecutionState* state,
                                   std::string* error_message) {
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

  if (instruction.opcode == "V_MOV_B32") {
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
      if (!WriteVectorOperand(instruction.operands[0], lane_index, value, state,
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
                                    std::string* error_message) {
  switch (instruction.opcode) {
    case Gfx1201CompiledOpcode::kSEndpgm:
    case Gfx1201CompiledOpcode::kSNop:
    case Gfx1201CompiledOpcode::kSMovB32:
    case Gfx1201CompiledOpcode::kSMovkI32:
    case Gfx1201CompiledOpcode::kVMovB32:
      return ExecuteDecodedSeedInstruction(instruction.decoded_instruction, state,
                                           error_message);
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
  return FindGfx1201InstructionSupport(opcode) != nullptr &&
         IsExecutableSeedOpcode(opcode);
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
    if (FindGfx1201InstructionSupport(instruction.opcode) == nullptr) {
      if (error_message != nullptr) {
        *error_message = "unknown gfx1201 opcode";
      }
      return false;
    }

    Gfx1201CompiledInstruction compiled_instruction;
    compiled_instruction.decoded_instruction = instruction;
    if (!TryCompileExecutableOpcode(instruction.opcode, &compiled_instruction.opcode)) {
      if (error_message != nullptr) {
        *error_message = BuildInterpreterBringupMessage(instruction.opcode);
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
    if (!ExecuteCompiledSeedInstruction(instruction, state, error_message)) {
      return false;
    }
    if (!state->halted) {
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
