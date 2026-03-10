#include "lib/sim/isa/gfx1201/interpreter.h"

#include <string>

namespace mirage::sim::isa {
namespace {

std::string BuildInterpreterBringupMessage() {
  std::string message =
      "gfx1201 interpreter scaffold only; carry-over families:";
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

template <typename ProgramT>
bool ReturnInterpreterScaffoldError(ProgramT program,
                                    WaveExecutionState* state,
                                    std::string* error_message) {
  if (state == nullptr) {
    if (error_message != nullptr) {
      *error_message = "wave execution state must not be null";
    }
    return false;
  }
  if (program.empty()) {
    return true;
  }
  if (error_message != nullptr) {
    *error_message = BuildInterpreterBringupMessage();
  }
  return false;
}

}  // namespace

bool Gfx1201Interpreter::Supports(std::string_view opcode) const {
  (void)opcode;
  return false;
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
  if (program.empty()) {
    return true;
  }

  if (error_message != nullptr) {
    *error_message = BuildInterpreterBringupMessage();
  }
  return false;
}

bool Gfx1201Interpreter::ExecuteProgram(std::span<const DecodedInstruction> program,
                                        WaveExecutionState* state,
                                        std::string* error_message) const {
  return ReturnInterpreterScaffoldError(program, state, error_message);
}

bool Gfx1201Interpreter::ExecuteProgram(std::span<const DecodedInstruction> program,
                                        WaveExecutionState* state,
                                        ExecutionMemory* memory,
                                        std::string* error_message) const {
  (void)memory;
  return ReturnInterpreterScaffoldError(program, state, error_message);
}

bool Gfx1201Interpreter::ExecuteProgram(
    std::span<const Gfx1201CompiledInstruction> program,
    WaveExecutionState* state,
    std::string* error_message) const {
  return ReturnInterpreterScaffoldError(program, state, error_message);
}

bool Gfx1201Interpreter::ExecuteProgram(
    std::span<const Gfx1201CompiledInstruction> program,
    WaveExecutionState* state,
    ExecutionMemory* memory,
    std::string* error_message) const {
  (void)memory;
  return ReturnInterpreterScaffoldError(program, state, error_message);
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
