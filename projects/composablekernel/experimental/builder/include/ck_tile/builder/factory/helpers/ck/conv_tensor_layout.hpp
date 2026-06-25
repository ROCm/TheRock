// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/tuple.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/builder_utils.hpp"

namespace ck_tile::builder::factory::internal {

template <TensorLayout Layout>
struct LayoutToCK
{
    static_assert(sizeof(UnsupportedEnumValue<Layout>) == 0,
                  "Unsupported layout conversion to CK.");
};

// Bias layouts
template <>
struct LayoutToCK<TensorLayout::G_K_strided>
{
    using type = ck::tensor_layout::convolution::G_K;
};
template <>
struct LayoutToCK<TensorLayout::GC>
{
    using type = ck::tensor_layout::convolution::GC;
};
template <>
struct LayoutToCK<TensorLayout::G_C_strided>
{
    using type = ck::tensor_layout::convolution::G_C;
};

// Input 1D
template <>
struct LayoutToCK<TensorLayout::NWGC>
{
    using type = ck::tensor_layout::convolution::NWGC;
};
template <>
struct LayoutToCK<TensorLayout::NGCW>
{
    using type = ck::tensor_layout::convolution::NGCW;
};
template <>
struct LayoutToCK<TensorLayout::GNWC>
{
    using type = ck::tensor_layout::convolution::GNWC;
};

// Input 2D
template <>
struct LayoutToCK<TensorLayout::NGCHW>
{
    using type = ck::tensor_layout::convolution::NGCHW;
};
template <>
struct LayoutToCK<TensorLayout::NHWGC>
{
    using type = ck::tensor_layout::convolution::NHWGC;
};
template <>
struct LayoutToCK<TensorLayout::GNHWC>
{
    using type = ck::tensor_layout::convolution::GNHWC;
};

// Input 3D
template <>
struct LayoutToCK<TensorLayout::NGCDHW>
{
    using type = ck::tensor_layout::convolution::NGCDHW;
};
template <>
struct LayoutToCK<TensorLayout::NDHWGC>
{
    using type = ck::tensor_layout::convolution::NDHWGC;
};
template <>
struct LayoutToCK<TensorLayout::GNDHWC>
{
    using type = ck::tensor_layout::convolution::GNDHWC;
};

// Weight 1D
template <>
struct LayoutToCK<TensorLayout::GKXC>
{
    using type = ck::tensor_layout::convolution::GKXC;
};
template <>
struct LayoutToCK<TensorLayout::GKCX>
{
    using type = ck::tensor_layout::convolution::GKCX;
};

// Weight 2D
template <>
struct LayoutToCK<TensorLayout::GKYXC>
{
    using type = ck::tensor_layout::convolution::GKYXC;
};
template <>
struct LayoutToCK<TensorLayout::GKCYX>
{
    using type = ck::tensor_layout::convolution::GKCYX;
};

// Weight 3D
template <>
struct LayoutToCK<TensorLayout::GKCZYX>
{
    using type = ck::tensor_layout::convolution::GKCZYX;
};
template <>
struct LayoutToCK<TensorLayout::GKZYXC>
{
    using type = ck::tensor_layout::convolution::GKZYXC;
};

// Output 1D
template <>
struct LayoutToCK<TensorLayout::NWGK>
{
    using type = ck::tensor_layout::convolution::NWGK;
};
template <>
struct LayoutToCK<TensorLayout::NGKW>
{
    using type = ck::tensor_layout::convolution::NGKW;
};
template <>
struct LayoutToCK<TensorLayout::GNWK>
{
    using type = ck::tensor_layout::convolution::GNWK;
};

// Output 2D
template <>
struct LayoutToCK<TensorLayout::NGKHW>
{
    using type = ck::tensor_layout::convolution::NGKHW;
};
template <>
struct LayoutToCK<TensorLayout::NHWGK>
{
    using type = ck::tensor_layout::convolution::NHWGK;
};
template <>
struct LayoutToCK<TensorLayout::GNHWK>
{
    using type = ck::tensor_layout::convolution::GNHWK;
};

// Output 3D
template <>
struct LayoutToCK<TensorLayout::NGKDHW>
{
    using type = ck::tensor_layout::convolution::NGKDHW;
};
template <>
struct LayoutToCK<TensorLayout::NDHWGK>
{
    using type = ck::tensor_layout::convolution::NDHWGK;
};
template <>
struct LayoutToCK<TensorLayout::GNDHWK>
{
    using type = ck::tensor_layout::convolution::GNDHWK;
};

template <TensorLayout Layout>
consteval auto TensorLayoutToCK()
{
    return typename LayoutToCK<Layout>::type{};
}

struct EmptyAuxiliaryTensorLayout
{
    using type = ck::Tuple<>;
};

template <auto AuxiliaryTensorConfigsArray, size_t... Indices>
consteval auto GetAuxiliaryTensorLayoutTuple(std::index_sequence<Indices...>)
{
    return ck::Tuple<
        decltype(TensorLayoutToCK<AuxiliaryTensorConfigsArray[Indices].layout>())...>{};
}

template <auto AuxiliaryTensorConfigsValue, size_t SPATIAL_DIM, ConvDirection DIR>
    requires(ConvSpatialDim<SPATIAL_DIM>)
struct AuxiliaryTensorLayouts
{
    static constexpr auto Size = AuxiliaryTensorConfigsValue.size();
    using type = decltype(GetAuxiliaryTensorLayoutTuple<AuxiliaryTensorConfigsValue>(
        std::make_index_sequence<Size>{}));
};

// TODO: Currently only the ouput tensor can have auxiliary tensors (e.g., bias).
template <auto Signature, size_t SPATIAL_DIM, ConvDirection DIR>
    requires(HasElementwiseOpWithAuxiliaryOperands<decltype(Signature.output)>)
consteval auto GetAuxiliaryTensorLayouts()
{
    return AuxiliaryTensorLayouts<Signature.output.operation.auxiliary_operand_configs,
                                  SPATIAL_DIM,
                                  DIR>{};
}

template <auto Signature, size_t SPATIAL_DIM, ConvDirection DIR>
    requires(!HasElementwiseOpWithAuxiliaryOperands<decltype(Signature.output)>)
consteval auto GetAuxiliaryTensorLayouts()
{
    return EmptyAuxiliaryTensorLayout{};
}

template <auto Signature, size_t SPATIAL_DIM, ConvDirection DIR>
    requires(ConvSpatialDim<SPATIAL_DIM> &&
             ValidConvInputLayoutForSpatialDim<Signature.input.config.layout, SPATIAL_DIM> &&
             ValidConvWeightLayoutForSpatialDim<Signature.weight.config.layout, SPATIAL_DIM> &&
             ValidConvOutputLayoutForSpatialDim<Signature.output.config.layout, SPATIAL_DIM>)
struct ConvTensorLayouts
{
    static_assert(DIR == ConvDirection::FORWARD, "Only Forward convolution is supported.");
    using ALayout  = decltype(TensorLayoutToCK<Signature.input.config.layout>());
    using BLayout  = decltype(TensorLayoutToCK<Signature.weight.config.layout>());
    using ELayout  = decltype(TensorLayoutToCK<Signature.output.config.layout>());
    using DsLayout = decltype(GetAuxiliaryTensorLayouts<Signature, SPATIAL_DIM, DIR>())::type;
};

} // namespace ck_tile::builder::factory::internal
