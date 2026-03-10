#ifndef MIRAGE_SIM_ISA_GFX1201_INTERPRETER_H_
#define MIRAGE_SIM_ISA_GFX1201_INTERPRETER_H_

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/common/execution_memory.h"
#include "lib/sim/isa/common/wave_execution_state.h"
#include "lib/sim/isa/gfx1201/architecture_profile.h"

namespace mirage::sim::isa {

struct Gfx1201CompiledInstruction {
  DecodedInstruction decoded_instruction;
};

class Gfx1201Interpreter {
 public:
  bool Supports(std::string_view opcode) const;
  bool CompileProgram(std::span<const DecodedInstruction> program,
                      std::vector<Gfx1201CompiledInstruction>* compiled_program,
                      std::string* error_message = nullptr) const;
  bool ExecuteProgram(std::span<const DecodedInstruction> program,
                      WaveExecutionState* state,
                      std::string* error_message = nullptr) const;
  bool ExecuteProgram(std::span<const DecodedInstruction> program,
                      WaveExecutionState* state,
                      ExecutionMemory* memory,
                      std::string* error_message) const;
  bool ExecuteProgram(std::span<const Gfx1201CompiledInstruction> program,
                      WaveExecutionState* state,
                      std::string* error_message = nullptr) const;
  bool ExecuteProgram(std::span<const Gfx1201CompiledInstruction> program,
                      WaveExecutionState* state,
                      ExecutionMemory* memory,
                      std::string* error_message) const;

  std::span<const Gfx1201FamilyFocus> CarryOverFamilyFocus() const;
  std::span<const Gfx1201FamilyFocus> Rdna4DeltaFamilyFocus() const;
  std::string_view BringupStatus() const;
};

}  // namespace mirage::sim::isa

#endif  // MIRAGE_SIM_ISA_GFX1201_INTERPRETER_H_
