#include "lib/sim/isa/gfx1201/binary_decoder.h"

#include <sstream>
#include <string>

namespace mirage::sim::isa {
namespace {

std::string BuildDecoderBringupMessage() {
  std::string message =
      "gfx1201 decoder scaffold only; phase-0 compute seed encodings:";
  bool first = true;
  for (const Gfx1201DecoderSeedEncoding& encoding :
       GetGfx1201Phase0ComputeDecoderSeeds()) {
    message.append(first ? " " : ", ");
    message.append(encoding.encoding_name);
    first = false;
  }
  return message;
}

std::string BuildRouteMessage(const Gfx1201OpcodeRoute& route) {
  std::ostringstream stream;
  stream << "gfx1201 decoder stub routed phase-0 compute opcode to "
         << route.selector_rule->encoding_name << " opcode " << route.opcode;
  if (route.seed_entry != nullptr) {
    stream << " (" << route.seed_entry->instruction_name << ")";
  }
  if (route.status == Gfx1201OpcodeRouteStatus::kNeedsMoreWords) {
    stream << " but needs " << route.words_required << " dwords";
  } else if (route.status == Gfx1201OpcodeRouteStatus::kMatchedEncodingOnly) {
    stream << " with no matching seed entry";
  }
  return stream.str();
}

}  // namespace

bool Gfx1201BinaryDecoder::DecodeInstruction(
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

  *instruction = DecodedInstruction{};
  *words_consumed = 0;

  if (words.empty()) {
    if (error_message != nullptr) {
      *error_message = "instruction stream is empty";
    }
    return false;
  }

  Gfx1201OpcodeRoute route;
  if (SelectPhase0ComputeRoute(words, &route, error_message)) {
    if (error_message != nullptr) {
      *error_message = BuildRouteMessage(route);
    }
    return false;
  }

  if (error_message != nullptr) {
    *error_message = BuildDecoderBringupMessage();
  }
  return false;
}

bool Gfx1201BinaryDecoder::DecodeProgram(std::span<const std::uint32_t> words,
                                         std::vector<DecodedInstruction>* program,
                                         std::string* error_message) const {
  if (program == nullptr) {
    if (error_message != nullptr) {
      *error_message = "decoded program output must not be null";
    }
    return false;
  }

  program->clear();
  if (words.empty()) {
    return true;
  }

  Gfx1201OpcodeRoute route;
  if (SelectPhase0ComputeRoute(words, &route, error_message)) {
    if (error_message != nullptr) {
      *error_message = BuildRouteMessage(route);
    }
    return false;
  }

  if (error_message != nullptr) {
    *error_message = BuildDecoderBringupMessage();
  }
  return false;
}

bool Gfx1201BinaryDecoder::SelectPhase0ComputeRoute(
    std::span<const std::uint32_t> words,
    Gfx1201OpcodeRoute* route,
    std::string* error_message) const {
  return SelectGfx1201Phase0ComputeOpcodeRoute(words, route, error_message);
}

std::span<const Gfx1201DecoderSeedEncoding>
Gfx1201BinaryDecoder::Phase0ComputeSeeds() const {
  return GetGfx1201Phase0ComputeDecoderSeeds();
}

std::span<const Gfx1201OpcodeSelectorRule>
Gfx1201BinaryDecoder::Phase0ComputeSelectorRules() const {
  return GetGfx1201Phase0ComputeOpcodeSelectorRules();
}

const Gfx1201DecoderSeedEncoding* Gfx1201BinaryDecoder::FindPhase0ComputeSeed(
    std::string_view encoding_name) const {
  return FindGfx1201Phase0ComputeDecoderSeed(encoding_name);
}

std::span<const Gfx1201EncodingFocus> Gfx1201BinaryDecoder::Phase0EncodingFocus()
    const {
  return GetGfx1201Phase0DecoderFocus();
}

std::span<const Gfx1201EncodingFocus> Gfx1201BinaryDecoder::Phase1EncodingFocus()
    const {
  return GetGfx1201Phase1DecoderFocus();
}

std::string_view Gfx1201BinaryDecoder::BringupStatus() const {
  return DescribeGfx1201BringupPhase();
}

}  // namespace mirage::sim::isa
