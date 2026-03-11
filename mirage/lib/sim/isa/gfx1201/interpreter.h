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

enum class Gfx1201CompiledOpcode : std::uint8_t {
  kUnknown,
  kSEndpgm,
  kSNop,
  kSAddU32,
  kSAddI32,
  kSSubU32,
  kSCmpEqI32,
  kSCmpLgI32,
  kSCmpGtI32,
  kSCmpEqU32,
  kSCmpLgU32,
  kSCmpGeI32,
  kSCmpLtI32,
  kSCmpLeI32,
  kSCmpGtU32,
  kSCmpGeU32,
  kSCmpLtU32,
  kSCmpLeU32,
  kSBranch,
  kSCbranchScc0,
  kSCbranchScc1,
  kSCbranchVccz,
  kSCbranchVccnz,
  kSCbranchExecz,
  kSCbranchExecnz,
  kSMovB32,
  kSMovkI32,
  kVMovB32,
  kVCmpEqI32,
  kVCmpNeI32,
  kVCmpLtI32,
  kVCmpLeI32,
  kVCmpGtI32,
  kVCmpGeI32,
  kVCmpEqU32,
  kVCmpNeU32,
  kVCmpLtU32,
  kVCmpLeU32,
  kVCmpGtU32,
  kVCmpGeU32,
  kVNotB32,
  kVBfrevB32,
  kVCvtF32Ubyte0,
  kVCvtF32Ubyte1,
  kVCvtF32Ubyte2,
  kVCvtF32Ubyte3,
  kVCvtF32I32,
  kVCvtF32U32,
  kVCvtU32F32,
  kVCvtI32F32,
  kVAddU32,
  kVSubU32,
  kVSubrevU32,
  kVMinI32,
  kVMaxI32,
  kVMinU32,
  kVMaxU32,
  kVLshrrevB32,
  kVAshrrevI32,
  kVLshlrevB32,
  kVAndB32,
  kVOrB32,
  kVXorB32,
};

struct Gfx1201CompiledInstruction {
  Gfx1201CompiledOpcode opcode = Gfx1201CompiledOpcode::kUnknown;
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

  std::span<const std::string_view> ExecutableSeedOpcodes() const;
  std::span<const Gfx1201FamilyFocus> CarryOverFamilyFocus() const;
  std::span<const Gfx1201FamilyFocus> Rdna4DeltaFamilyFocus() const;
  std::string_view BringupStatus() const;
};

}  // namespace mirage::sim::isa

#endif  // MIRAGE_SIM_ISA_GFX1201_INTERPRETER_H_
