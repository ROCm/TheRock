#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "lib/sim/isa/common/decoded_instruction.h"
#include "lib/sim/isa/common/wave_execution_state.h"
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

constexpr std::uint32_t MakeSop1(std::uint32_t op,
                                 std::uint32_t sdst,
                                 std::uint32_t ssrc0) {
  std::uint32_t word = 0;
  word = SetBits(word, 0x17d, 23, 9);
  word = SetBits(word, sdst, 16, 7);
  word = SetBits(word, op, 8, 8);
  word = SetBits(word, ssrc0, 0, 8);
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

constexpr std::uint32_t MakeVop1(std::uint32_t op,
                                 std::uint32_t vdst,
                                 std::uint32_t src0) {
  std::uint32_t word = 0;
  word = SetBits(word, 0x3f, 25, 7);
  word = SetBits(word, vdst, 17, 8);
  word = SetBits(word, op, 9, 8);
  word = SetBits(word, src0, 0, 9);
  return word;
}

bool ExpectUnaryMove(const mirage::sim::isa::DecodedInstruction& instruction,
                     std::string_view expected_opcode,
                     mirage::sim::isa::OperandKind dst_kind,
                     std::uint16_t dst_index,
                     mirage::sim::isa::OperandKind src_kind,
                     std::uint32_t src_value_or_index) {
  using namespace mirage::sim::isa;
  if (instruction.opcode != expected_opcode || instruction.operand_count != 2u) {
    return false;
  }
  if (instruction.operands[0].kind != dst_kind ||
      instruction.operands[0].index != dst_index ||
      instruction.operands[1].kind != src_kind) {
    return false;
  }
  if (src_kind == OperandKind::kImm32) {
    return instruction.operands[1].imm32 == src_value_or_index;
  }
  return instruction.operands[1].index ==
         static_cast<std::uint16_t>(src_value_or_index);
}

bool ExpectSeedProgramState(const mirage::sim::isa::WaveExecutionState& state) {
  return state.sgprs[1] == 0xfffffffdu && state.sgprs[2] == 0xfffffffdu &&
         state.vgprs[3][0] == 0xfffffffdu && state.vgprs[3][1] == 0xfffffffdu &&
         state.vgprs[3][2] == 0xfeedfaceu && state.vgprs[3][3] == 0xfffffffdu &&
         state.halted && !state.waiting_on_barrier && state.pc == 4u;
}

}  // namespace

int main() {
  using namespace mirage::sim::isa;

  Gfx1201BinaryDecoder decoder;
  Gfx1201Interpreter interpreter;
  std::string error_message;
  DecodedInstruction instruction;
  std::size_t words_consumed = 0;

  const std::array<std::uint32_t, 2> scalar_literal_words{
      MakeSop1(0u, 7u, 255u), 0x12345678u};
  if (!Expect(decoder.DecodeInstruction(scalar_literal_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_MOV_B32 literal decode success") ||
      !Expect(words_consumed == 2u, "expected literal decode to consume 2 dwords") ||
      !Expect(ExpectUnaryMove(instruction, "S_MOV_B32", OperandKind::kSgpr, 7u,
                              OperandKind::kImm32, 0x12345678u),
              "expected decoded S_MOV_B32 literal operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> movk_words{MakeSopk(0u, 4u, 0xfffeu)};
  if (!Expect(decoder.DecodeInstruction(movk_words, &instruction, &words_consumed,
                                        &error_message),
              "expected S_MOVK_I32 decode success") ||
      !Expect(words_consumed == 1u, "expected S_MOVK_I32 to consume 1 dword") ||
      !Expect(ExpectUnaryMove(instruction, "S_MOVK_I32", OperandKind::kSgpr, 4u,
                              OperandKind::kImm32, 0xfffffffeu),
              "expected decoded S_MOVK_I32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> vector_move_words{MakeVop1(1u, 3u, 2u)};
  if (!Expect(decoder.DecodeInstruction(vector_move_words, &instruction,
                                        &words_consumed, &error_message),
              "expected V_MOV_B32 decode success") ||
      !Expect(words_consumed == 1u, "expected V_MOV_B32 to consume 1 dword") ||
      !Expect(ExpectUnaryMove(instruction, "V_MOV_B32", OperandKind::kVgpr, 3u,
                              OperandKind::kSgpr, 2u),
              "expected decoded V_MOV_B32 operands")) {
    return 1;
  }

  const std::array<std::uint32_t, 5> seed_program_words{
      MakeSopk(0u, 1u, 0xfffdu),
      MakeSop1(0u, 2u, 1u),
      MakeVop1(1u, 3u, 2u),
      MakeSopp(0u, 5u),
      MakeSopp(48u),
  };
  std::vector<DecodedInstruction> decoded_program;
  if (!Expect(decoder.DecodeProgram(seed_program_words, &decoded_program,
                                    &error_message),
              "expected phase-0 seed program decode success") ||
      !Expect(decoded_program.size() == 5u,
              "expected five decoded instructions") ||
      !Expect(decoded_program[0].opcode == "S_MOVK_I32",
              "expected decoded S_MOVK_I32") ||
      !Expect(decoded_program[1].opcode == "S_MOV_B32",
              "expected decoded S_MOV_B32") ||
      !Expect(decoded_program[2].opcode == "V_MOV_B32",
              "expected decoded V_MOV_B32") ||
      !Expect(decoded_program[3].opcode == "S_NOP", "expected decoded S_NOP") ||
      !Expect(decoded_program[4].opcode == "S_ENDPGM",
              "expected decoded S_ENDPGM")) {
    return 1;
  }

  WaveExecutionState decoded_state;
  decoded_state.exec_mask = 0xbu;
  decoded_state.vgprs[3][2] = 0xfeedfaceu;
  if (!Expect(interpreter.ExecuteProgram(decoded_program, &decoded_state,
                                         &error_message),
              "expected decoded seed program execution success") ||
      !Expect(ExpectSeedProgramState(decoded_state),
              "expected decoded seed program state")) {
    return 1;
  }

  std::vector<Gfx1201CompiledInstruction> compiled_program;
  if (!Expect(interpreter.CompileProgram(decoded_program, &compiled_program,
                                         &error_message),
              "expected compiled seed program success") ||
      !Expect(compiled_program.size() == decoded_program.size(),
              "expected compiled instruction count") ||
      !Expect(compiled_program[0].opcode == Gfx1201CompiledOpcode::kSMovkI32,
              "expected compiled S_MOVK_I32 opcode") ||
      !Expect(compiled_program[1].opcode == Gfx1201CompiledOpcode::kSMovB32,
              "expected compiled S_MOV_B32 opcode") ||
      !Expect(compiled_program[2].opcode == Gfx1201CompiledOpcode::kVMovB32,
              "expected compiled V_MOV_B32 opcode") ||
      !Expect(compiled_program[3].opcode == Gfx1201CompiledOpcode::kSNop,
              "expected compiled S_NOP opcode") ||
      !Expect(compiled_program[4].opcode == Gfx1201CompiledOpcode::kSEndpgm,
              "expected compiled S_ENDPGM opcode")) {
    return 1;
  }

  WaveExecutionState compiled_state;
  compiled_state.exec_mask = 0xbu;
  compiled_state.vgprs[3][2] = 0xfeedfaceu;
  if (!Expect(interpreter.ExecuteProgram(compiled_program, &compiled_state,
                                         &error_message),
              "expected compiled seed program execution success") ||
      !Expect(ExpectSeedProgramState(compiled_state),
              "expected compiled seed program state")) {
    return 1;
  }

  const std::array<std::uint32_t, 1> scc_move_words{MakeSop1(0u, 5u, 253u)};
  if (!Expect(decoder.DecodeInstruction(scc_move_words, &instruction,
                                        &words_consumed, &error_message),
              "expected S_MOV_B32 SCC decode success")) {
    return 1;
  }

  WaveExecutionState scc_state;
  scc_state.scc = true;
  const std::array<DecodedInstruction, 1> scc_program{instruction};
  if (!Expect(interpreter.ExecuteProgram(scc_program, &scc_state, &error_message),
              "expected SCC seed execution success") ||
      !Expect(scc_state.sgprs[5] == 1u, "expected SCC special source read") ||
      !Expect(scc_state.pc == 1u, "expected single-instruction program advance") ||
      !Expect(!scc_state.halted, "expected SCC move program to remain running")) {
    return 1;
  }

  return 0;
}
