// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"

namespace ck_tile::builder::factory::internal {
using ALayout = ck_tile::tensor_layout::convolution::NWGC;
template <TensorLayout Layout>
struct LayoutToCKTile
{
    static_assert(sizeof(UnsupportedEnumValue<Layout>) == 0,
                  "Unsupported layout conversion to CK.");
};

// Bias layouts
template <>
struct LayoutToCKTile<TensorLayout::G_K_strided>
{
    using type = ck_tile::tensor_layout::convolution::G_K;
};
template <>
struct LayoutToCKTile<TensorLayout::GC>
{
    using type = ck_tile::tensor_layout::convolution::GC;
};
template <>
struct LayoutToCKTile<TensorLayout::G_C_strided>
{
    using type = ck_tile::tensor_layout::convolution::G_C;
};

// Input 1D
template <>
struct LayoutToCKTile<TensorLayout::NWGC>
{
    using type = ck_tile::tensor_layout::convolution::NWGC;
};
template <>
struct LayoutToCKTile<TensorLayout::GNWC>
{
    using type = ck_tile::tensor_layout::convolution::GNWC;
};

// Input 2D
template <>
struct LayoutToCKTile<TensorLayout::NHWGC>
{
    using type = ck_tile::tensor_layout::convolution::NHWGC;
};
template <>
struct LayoutToCKTile<TensorLayout::GNHWC>
{
    using type = ck_tile::tensor_layout::convolution::GNHWC;
};

// Input 3D
template <>
struct LayoutToCKTile<TensorLayout::NDHWGC>
{
    using type = ck_tile::tensor_layout::convolution::NDHWGC;
};
template <>
struct LayoutToCKTile<TensorLayout::GNDHWC>
{
    using type = ck_tile::tensor_layout::convolution::GNDHWC;
};

// Weight 1D
template <>
struct LayoutToCKTile<TensorLayout::GKXC>
{
    using type = ck_tile::tensor_layout::convolution::GKXC;
};
template <>
struct LayoutToCKTile<TensorLayout::GKCX>
{
    using type = ck_tile::tensor_layout::convolution::GKCX;
};

// Weight 2D
template <>
struct LayoutToCKTile<TensorLayout::GKYXC>
{
    using type = ck_tile::tensor_layout::convolution::GKYXC;
};
template <>
struct LayoutToCKTile<TensorLayout::GKCYX>
{
    using type = ck_tile::tensor_layout::convolution::GKCYX;
};

// Weight 3D
template <>
struct LayoutToCKTile<TensorLayout::GKCZYX>
{
    using type = ck_tile::tensor_layout::convolution::GKCZYX;
};
template <>
struct LayoutToCKTile<TensorLayout::GKZYXC>
{
    using type = ck_tile::tensor_layout::convolution::GKZYXC;
};

// Output 1D
template <>
struct LayoutToCKTile<TensorLayout::NWGK>
{
    using type = ck_tile::tensor_layout::convolution::NWGK;
};
template <>
struct LayoutToCKTile<TensorLayout::GNWK>
{
    using type = ck_tile::tensor_layout::convolution::GNWK;
};

// Output 2D
template <>
struct LayoutToCKTile<TensorLayout::NHWGK>
{
    using type = ck_tile::tensor_layout::convolution::NHWGK;
};
template <>
struct LayoutToCKTile<TensorLayout::GNHWK>
{
    using type = ck_tile::tensor_layout::convolution::GNHWK;
};

// Output 3D
template <>
struct LayoutToCKTile<TensorLayout::NDHWGK>
{
    using type = ck_tile::tensor_layout::convolution::NDHWGK;
};
template <>
struct LayoutToCKTile<TensorLayout::GNDHWK>
{
    using type = ck_tile::tensor_layout::convolution::GNDHWK;
};

template <TensorLayout Layout>
consteval auto TensorLayoutToCKTile()
{
    return typename LayoutToCKTile<Layout>::type{};
}

struct EmptyAuxiliaryTileTensorLayout
{
    using type = ck_tile::tuple<>;
};

template <auto AuxiliaryTileTensorConfigsArray, size_t... Indices>
consteval auto GetAuxiliaryTileTensorLayoutTuple(std::index_sequence<Indices...>)
{
    return ck_tile::tuple<
        decltype(TensorLayoutToCKTile<AuxiliaryTileTensorConfigsArray[Indices].layout>())...>{};
}

template <auto AuxiliaryTileTensorConfigsValue, size_t SPATIAL_DIM>
    requires(ConvSpatialDim<SPATIAL_DIM>)
struct AuxiliaryTileTensorLayouts
{
    static constexpr auto Size = AuxiliaryTileTensorConfigsValue.size();
    using type = decltype(GetAuxiliaryTileTensorLayoutTuple<AuxiliaryTileTensorConfigsValue>(
        std::make_index_sequence<Size>{}));
};

// TODO: Currently only the ouput tensor can have auxiliary tensors (e.g., bias).
template <auto Signature, size_t SPATIAL_DIM>
    requires(HasElementwiseOpWithAuxiliaryOperands<decltype(Signature.output)>)
consteval auto GetAuxiliaryTileTensorLayouts()
{
    return AuxiliaryTileTensorLayouts<Signature.output.operation.auxiliary_operand_configs,
                                      SPATIAL_DIM>{};
}

template <auto Signature, size_t SPATIAL_DIM>
    requires(!HasElementwiseOpWithAuxiliaryOperands<decltype(Signature.output)>)
consteval auto GetAuxiliaryTileTensorLayouts()
{
    return EmptyAuxiliaryTileTensorLayout{};
}

template <auto Signature, size_t SPATIAL_DIM>
    requires(ConvSpatialDim<SPATIAL_DIM> &&
             ValidConvInputLayoutForSpatialDim<Signature.input.config.layout, SPATIAL_DIM> &&
             ValidConvWeightLayoutForSpatialDim<Signature.weight.config.layout, SPATIAL_DIM> &&
             ValidConvOutputLayoutForSpatialDim<Signature.output.config.layout, SPATIAL_DIM>)
struct TileConvTensorLayouts
{
    using ALayout  = decltype(TensorLayoutToCKTile<Signature.input.config.layout>());
    using BLayout  = decltype(TensorLayoutToCKTile<Signature.weight.config.layout>());
    using ELayout  = decltype(TensorLayoutToCKTile<Signature.output.config.layout>());
    using DsLayout = decltype(GetAuxiliaryTileTensorLayouts<Signature, SPATIAL_DIM>())::type;
};

} // namespace ck_tile::builder::factory::internal
