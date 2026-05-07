// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// This file defines the compile-time "signature" for grouped convolution operations.
// A signature is a collection of properties that fully describe a convolution kernel's
// mathematical characteristics. It uses C++20 concepts and enums to specify these
// properties, enabling compile-time validation and specialization.
//
// The core components of a signature are:
//   - Spatial dimensionality (1D, 2D, 3D)
//   - Operational direction (Forward, Backward Data, Backward Weight)
//   - Tensor memory layout (Channels First/Last)
//   - Data type (FP32, FP16, BF16)
//   - Fused element-wise operation (e.g., Bias, Clamp)
//
// The file also provides predicate concepts to query the properties of a given
// signature at compile time.
#pragma once

#include <concepts>
#include <type_traits>

#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder {

// Constrains convolution to 1D, 2D, or 3D spatial dimensions.
template <auto N>
concept ConvSpatialDim = std::is_integral_v<decltype(N)> && (N == 1 || N == 2 || N == 3);

// Constrains convolution data types to common floating-point types.
template <DataType T>
concept ValidConvDataType =
    (T == DataType::FP32) || (T == DataType::FP16) || (T == DataType::BF16) ||
    (T == DataType::FP8) || (T == DataType::I8) || (T == DataType::U8);

template <TensorLayout L>
concept BiasTensorLayout =
    (L == TensorLayout::GC) || (L == TensorLayout::G_C_strided) || (L == TensorLayout::G_K_strided);

template <TensorLayout L>
concept ConvInputLayout1D =
    (L == TensorLayout::GNCW) || (L == TensorLayout::GNWC) || (L == TensorLayout::NWGC) ||
    (L == TensorLayout::NGCW) || (L == TensorLayout::G_NW_C_strided);

template <TensorLayout L>
concept ConvInputLayout2D =
    (L == TensorLayout::GNCHW) || (L == TensorLayout::GNHWC) || (L == TensorLayout::NHWGC) ||
    (L == TensorLayout::NGCHW) || (L == TensorLayout::G_NHW_C_strided);

template <TensorLayout L>
concept ConvInputLayout3D =
    (L == TensorLayout::GNCDHW) || (L == TensorLayout::GNDHWC) || (L == TensorLayout::NDHWGC) ||
    (L == TensorLayout::NGCDHW) || (L == TensorLayout::G_NDHW_C_strided);

template <TensorLayout L>
concept ConvWeightLayout1D = (L == TensorLayout::GKXC) || (L == TensorLayout::GKCX) ||
                             (L == TensorLayout::KXGC) || (L == TensorLayout::G_K_X_C_strided);

template <TensorLayout L>
concept ConvWeightLayout2D = (L == TensorLayout::GKYXC) || (L == TensorLayout::GKCYX) ||
                             (L == TensorLayout::KYXGC) || (L == TensorLayout::G_K_YX_C_strided);

template <TensorLayout L>
concept ConvWeightLayout3D = (L == TensorLayout::GKZYXC) || (L == TensorLayout::GKCZYX) ||
                             (L == TensorLayout::KZYXGC) || (L == TensorLayout::G_K_ZYX_C_strided);

template <TensorLayout L>
concept ConvOutputLayout1D =
    (L == TensorLayout::GNKW) || (L == TensorLayout::GNWK) || (L == TensorLayout::NWGK) ||
    (L == TensorLayout::NGKW) || (L == TensorLayout::G_NW_K_strided);

template <TensorLayout L>
concept ConvOutputLayout2D =
    (L == TensorLayout::GNKHW) || (L == TensorLayout::GNHWK) || (L == TensorLayout::NHWGK) ||
    (L == TensorLayout::NGKHW) || (L == TensorLayout::G_NHW_K_strided);

template <TensorLayout L>
concept ConvOutputLayout3D =
    (L == TensorLayout::GNKDHW) || (L == TensorLayout::GNDHWK) || (L == TensorLayout::NDHWGK) ||
    (L == TensorLayout::NGKDHW) || (L == TensorLayout::G_NDHW_K_strided);

template <typename T>
concept HasDataType = requires(T t) {
    { t.data_type };
};

// Note: for signature and TensorConfigDescriptor,
// it is not required to provide a default data type, but if one is provided, check if well defined
template <typename T>
concept DataTypeWellDefinedIfProvided = requires(T t) {
    requires !HasDataType<T> || requires {
        { t.data_type } -> std::convertible_to<DataType>;
    };
};

template <typename T>
concept TensorConfigDescriptor = requires(T t) {
    { t.layout } -> std::convertible_to<TensorLayout>;
    requires DataTypeWellDefinedIfProvided<T>;
};

template <typename T>
concept HasAuxiliaryOperandConfigs = requires(T t) {
    { t.auxiliary_operand_configs };
};

namespace detail {
template <typename T>
struct IsArrayOfTensorConfigDescriptors : std::false_type
{
};

template <typename T, std::size_t N>
    requires TensorConfigDescriptor<T>
struct IsArrayOfTensorConfigDescriptors<std::array<T, N>> : std::true_type
{
};
} // namespace detail

template <typename T>
concept ConvertibleToArrayOfTensorConfigs =
    detail::IsArrayOfTensorConfigDescriptors<std::remove_cvref_t<T>>::value;

template <typename T>
concept AuxiliaryOperandConfigsWellDefinedIfProvided = requires(T t) {
    requires !HasAuxiliaryOperandConfigs<T> || requires {
        { t.auxiliary_operand_configs } -> ConvertibleToArrayOfTensorConfigs;
    };
};

template <typename T>
concept TensorOperatorDescriptor = requires(T t) {
    { t.elementwise_operation } -> std::convertible_to<ElementwiseOperation>;
    requires AuxiliaryOperandConfigsWellDefinedIfProvided<T>;
};

template <typename T>
concept HasTensorOp = requires(T t) {
    { t.operation };
};

template <typename T>
concept HasConvolutionDirection = requires(T t) {
    { t.direction };
};

// Note: it is not required to provide an ElementwiseOp, but if one is provided, check if well
// defined
template <typename T>
concept ElementwiseOpWellDefinedIfProvided =
    !HasTensorOp<T> || requires(T t) { requires TensorOperatorDescriptor<decltype(t.operation)>; };

// Note: it is not required to provide a convolution, but if one is provided, check if well defined
template <typename T>
concept ConvolutionDirectionWellDefinedIfProvided = requires(T t) {
    requires !HasConvolutionDirection<T> || requires {
        { t.direction } -> std::convertible_to<ConvDirection>;
    };
};

// Concept for the convolution tensor
template <typename T>
concept ConvTensorDescriptor = requires(T t) {
    { t.config } -> TensorConfigDescriptor;
    requires ElementwiseOpWellDefinedIfProvided<T>;
};

template <typename T>
concept HasElementwiseOpWithAuxiliaryOperands = requires(T t) {
    requires HasTensorOp<T>;
    requires HasAuxiliaryOperandConfigs<decltype(t.operation)>;
};

// Concept for a type that defines a convolution's operational signature.
template <typename T>
concept ConvSignatureDescriptor = requires(T t) {
    { t.spatial_dim } -> std::convertible_to<unsigned int>;
    { t.input } -> ConvTensorDescriptor;
    { t.weight } -> ConvTensorDescriptor;
    { t.output } -> ConvTensorDescriptor;
    requires ConvolutionDirectionWellDefinedIfProvided<T>;
    requires DataTypeWellDefinedIfProvided<T>;
};

// Concept to validate a convolution signature's values.
template <auto Sig>
concept ValidConvSignature = requires {
    requires ConvSpatialDim<Sig.spatial_dim>;
    requires ValidConvDataType<Sig.data_type>;
};

// Predicate for forward convolution (default if direction is not included).
template <auto Sig>
concept ConvDirectionIsForward =
    !requires { Sig.direction; } || (Sig.direction == ConvDirection::FORWARD);

// Predicate for backward data convolution.
template <auto Sig>
concept ConvDirectionIsBackwardData = (Sig.direction == ConvDirection::BACKWARD_DATA);

// Predicate for backward weight convolution.
template <auto Sig>
concept ConvDirectionIsBackwardWeight = (Sig.direction == ConvDirection::BACKWARD_WEIGHT);

// Constraints for forward convolution input layouts.
template <TensorLayout L, size_t SpatialDim>
concept ValidConvInputLayoutForSpatialDim =
    (SpatialDim == 1 && ConvInputLayout1D<L>) || (SpatialDim == 2 && ConvInputLayout2D<L>) ||
    (SpatialDim == 3 && ConvInputLayout3D<L>);

// Constraints for forward convolution output layouts.
template <TensorLayout L, size_t SpatialDim>
concept ValidConvOutputLayoutForSpatialDim =
    (SpatialDim == 1 && ConvOutputLayout1D<L>) || (SpatialDim == 2 && ConvOutputLayout2D<L>) ||
    (SpatialDim == 3 && ConvOutputLayout3D<L>);

// Constraints for forward convolution weight layouts.
template <TensorLayout L, size_t SpatialDim>
concept ValidConvWeightLayoutForSpatialDim =
    (SpatialDim == 1 && ConvWeightLayout1D<L>) || (SpatialDim == 2 && ConvWeightLayout2D<L>) ||
    (SpatialDim == 3 && ConvWeightLayout3D<L>);

} // namespace ck_tile::builder
