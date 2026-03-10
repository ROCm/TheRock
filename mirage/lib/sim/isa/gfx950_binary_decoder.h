#ifndef MIRAGE_SIM_ISA_GFX950_BINARY_DECODER_H_
#define MIRAGE_SIM_ISA_GFX950_BINARY_DECODER_H_

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "lib/sim/isa/decoded_instruction.h"

namespace mirage::sim::isa {

class Gfx950BinaryDecoder {
 public:
  bool DecodeInstruction(std::span<const std::uint32_t> words,
                         DecodedInstruction* instruction,
                         std::size_t* words_consumed,
                         std::string* error_message = nullptr) const;
  bool DecodeProgram(std::span<const std::uint32_t> words,
                     std::vector<DecodedInstruction>* program,
                     std::string* error_message = nullptr) const;

 private:
  const char* FindInstructionName(const char* encoding_name,
                                  std::uint32_t opcode) const;
  bool DecodeSopp(std::uint32_t word,
                  DecodedInstruction* instruction,
                  std::string* error_message) const;
  bool DecodeSmem(std::span<const std::uint32_t> words,
                  DecodedInstruction* instruction,
                  std::size_t* words_consumed,
                  std::string* error_message) const;
  bool DecodeFlat(std::span<const std::uint32_t> words,
                  DecodedInstruction* instruction,
                  std::size_t* words_consumed,
                  std::string* error_message) const;
  bool DecodeDs(std::span<const std::uint32_t> words,
                DecodedInstruction* instruction,
                std::size_t* words_consumed,
                std::string* error_message) const;
  bool DecodeFlatGlobal(std::span<const std::uint32_t> words,
                        DecodedInstruction* instruction,
                        std::size_t* words_consumed,
                        std::string* error_message) const;
  bool DecodeSopc(std::span<const std::uint32_t> words,
                  DecodedInstruction* instruction,
                  std::size_t* words_consumed,
                  std::string* error_message) const;
  bool DecodeSopk(std::span<const std::uint32_t> words,
                  DecodedInstruction* instruction,
                  std::size_t* words_consumed,
                  std::string* error_message) const;
  bool DecodeSop1(std::span<const std::uint32_t> words,
                  DecodedInstruction* instruction,
                  std::size_t* words_consumed,
                  std::string* error_message) const;
  bool DecodeSop2(std::span<const std::uint32_t> words,
                  DecodedInstruction* instruction,
                  std::size_t* words_consumed,
                  std::string* error_message) const;
  bool DecodeVop1(std::span<const std::uint32_t> words,
                  DecodedInstruction* instruction,
                  std::size_t* words_consumed,
                  std::string* error_message) const;
  bool DecodeVop2(std::span<const std::uint32_t> words,
                  DecodedInstruction* instruction,
                  std::size_t* words_consumed,
                  std::string* error_message) const;
  bool DecodeVopc(std::span<const std::uint32_t> words,
                  DecodedInstruction* instruction,
                  std::size_t* words_consumed,
                  std::string* error_message) const;
  bool DecodeVop3(std::span<const std::uint32_t> words,
                  DecodedInstruction* instruction,
                  std::size_t* words_consumed,
                  std::string* error_message) const;

  bool DecodeScalarDestination(std::uint32_t raw_value,
                               InstructionOperand* operand,
                               std::string* error_message) const;
  bool DecodeVectorDestination(std::uint32_t raw_value,
                               InstructionOperand* operand,
                               std::string* error_message) const;
  bool DecodeScalarSource(std::uint32_t raw_value,
                          std::span<const std::uint32_t> literal_words,
                          std::size_t* literal_words_consumed,
                          InstructionOperand* operand,
                          std::string* error_message) const;
  bool DecodeVectorSource(std::uint32_t raw_value,
                          std::span<const std::uint32_t> literal_words,
                          std::size_t* literal_words_consumed,
                          InstructionOperand* operand,
                          std::string* error_message) const;
  bool DecodeVectorRegisterSource(std::uint32_t raw_value,
                                  InstructionOperand* operand,
                                  std::string* error_message) const;
  bool DecodeSmemBase(std::uint32_t raw_value,
                      InstructionOperand* operand,
                      std::string* error_message) const;
  bool DecodeSmemOffset(std::uint64_t instruction_word,
                        InstructionOperand* operand,
                        std::string* error_message) const;
  bool DecodeFlatAddress(std::uint32_t raw_value,
                         InstructionOperand* operand,
                         std::string* error_message) const;
  bool DecodeFlatGlobalBase(std::uint32_t raw_value,
                            InstructionOperand* operand,
                            std::string* error_message) const;
};

}  // namespace mirage::sim::isa

#endif  // MIRAGE_SIM_ISA_GFX950_BINARY_DECODER_H_
