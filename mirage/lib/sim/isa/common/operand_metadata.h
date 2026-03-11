#ifndef MIRAGE_SIM_ISA_COMMON_OPERAND_METADATA_H_
#define MIRAGE_SIM_ISA_COMMON_OPERAND_METADATA_H_

#include <array>
#include <cstdint>

namespace mirage::sim::isa {

enum class OperandAccess {
  kRead,
  kWrite,
  kReadWrite,
};

enum class OperandRole {
  kUnknown,
  kDestination,
  kSource0,
  kSource1,
  kSource2,
  kAccumulator,
  kScale,
  kPairedScale,
  kTensorDescriptor,
  kTensorCoordinate,
  kLdsDestination,
  kLdsSource,
};

enum class OperandSlotKind {
  kUnknown,
  kDestination,
  kScalarDestination,
  kSource0,
  kSource1,
  kSource2,
  kAccumulatorSource,
  kScaleSource,
  kPairedScaleSource,
  kTensorDescriptorSource,
  kTensorCoordinateSource,
  kLdsDestination,
  kLdsSource,
};

enum class OperandValueClass {
  kUnknown,
  kVectorRegister,
  kScalarRegister,
  kPackedVector,
  kMatrixFragment,
  kAccumulatorFragment,
  kTensorDescriptor,
  kTensorCoordinate,
  kLdsAddress,
};

enum class FragmentKind {
  kUnknown,
  kScalar,
  kVector,
  kPacked,
  kMatrix,
  kTensorDescriptor,
  kTensorCoordinate,
  kAddress,
};

struct FragmentShape {
  FragmentKind kind = FragmentKind::kUnknown;
  std::uint16_t rows = 1;
  std::uint16_t columns = 1;
  std::uint16_t depth = 1;
  std::uint8_t element_bit_width = 0;
  std::uint8_t packed_elements = 0;
  std::uint8_t wave_size = 0;
};

constexpr FragmentShape MakeScalarFragmentShape(std::uint8_t element_bit_width) {
  return {FragmentKind::kScalar, 1, 1, 1, element_bit_width, 1, 0};
}

constexpr FragmentShape MakeVectorFragmentShape(std::uint32_t components,
                                                std::uint8_t element_bit_width) {
  return {FragmentKind::kVector,
          1,
          static_cast<std::uint16_t>(components),
          1,
          element_bit_width,
          static_cast<std::uint8_t>(components),
          0};
}

constexpr FragmentShape MakePackedFragmentShape(std::uint32_t packed_elements,
                                                std::uint8_t element_bit_width) {
  return {FragmentKind::kPacked,
          1,
          1,
          1,
          element_bit_width,
          static_cast<std::uint8_t>(packed_elements),
          0};
}

constexpr FragmentShape MakeMatrixFragmentShape(std::uint16_t rows,
                                                std::uint16_t columns,
                                                std::uint16_t depth,
                                                std::uint8_t element_bit_width,
                                                std::uint8_t wave_size) {
  return {FragmentKind::kMatrix,
          rows,
          columns,
          depth,
          element_bit_width,
          0,
          wave_size};
}

constexpr FragmentShape MakeTensorDescriptorFragmentShape() {
  return {FragmentKind::kTensorDescriptor, 1, 1, 1, 0, 1, 0};
}

constexpr FragmentShape MakeTensorCoordinateFragmentShape() {
  return {FragmentKind::kTensorCoordinate, 1, 1, 1, 0, 1, 0};
}

constexpr FragmentShape MakeAddressFragmentShape(std::uint8_t element_bit_width) {
  return {FragmentKind::kAddress, 1, 1, 1, element_bit_width, 1, 0};
}

struct OperandDescriptor {
  OperandRole role = OperandRole::kUnknown;
  OperandSlotKind slot_kind = OperandSlotKind::kUnknown;
  OperandValueClass value_class = OperandValueClass::kUnknown;
  OperandAccess access = OperandAccess::kRead;
  FragmentShape fragment_shape{};
  std::uint8_t component_count = 1;
  bool is_implicit = false;
};

struct OperandRoleBinding {
  OperandRole role = OperandRole::kUnknown;
  std::uint32_t count = 0;
  bool is_output = false;
  bool is_implicit = false;
};

struct OperandRoleRecord {
  std::array<OperandRoleBinding, 8> bindings{};
  std::uint32_t binding_count = 0;
};

struct OperandSlotBinding {
  OperandSlotKind slot_kind = OperandSlotKind::kUnknown;
  OperandValueClass value_class = OperandValueClass::kUnknown;
  std::uint32_t logical_operand_index = 0;
  std::uint32_t component_count = 0;
  bool is_output = false;
  bool is_implicit = false;
  FragmentShape fragment_shape{};
};

struct OperandSlotRecord {
  std::array<OperandSlotBinding, 8> bindings{};
  std::uint32_t binding_count = 0;
};

}  // namespace mirage::sim::isa

#endif  // MIRAGE_SIM_ISA_COMMON_OPERAND_METADATA_H_
