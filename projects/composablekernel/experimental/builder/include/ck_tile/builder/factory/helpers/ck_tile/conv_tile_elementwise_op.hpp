// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/builder/builder_utils.hpp"
#include "ck_tile/builder/types.hpp"

namespace ck_tile::builder::factory::internal {

template <ElementwiseOperation Op>
struct ElementwiseOpToCKTile
{
    static_assert(sizeof(UnsupportedEnumValue<Op>) == 0,
                  "Unsupported elementwise operation conversion to CK.");
};

template <>
struct ElementwiseOpToCKTile<ElementwiseOperation::PASS_THROUGH>
{
    using Op = ck_tile::element_wise::PassThrough;
};

template <>
struct ElementwiseOpToCKTile<ElementwiseOperation::SCALE>
{
    using Op = ck_tile::element_wise::Scale;
};

template <>
struct ElementwiseOpToCKTile<ElementwiseOperation::CLAMP>
{
    using Op = ck_tile::element_wise::Clamp;
};

template <auto TensorDesc>
consteval auto GetTileElementwiseOp()
{
    if constexpr(HasTensorOp<decltype(TensorDesc)>)
    {
        constexpr auto op = TensorDesc.operation.elementwise_operation;
        return ElementwiseOpToCKTile<op>{};
    }
    else
    {
        return ElementwiseOpToCKTile<ElementwiseOperation::PASS_THROUGH>{};
    }
}

template <auto Sig>
struct TileElementwiseOps
{
    static constexpr auto input_op  = GetTileElementwiseOp<Sig.input>();
    static constexpr auto weight_op = GetTileElementwiseOp<Sig.weight>();
    static constexpr auto output_op = GetTileElementwiseOp<Sig.output>();
    using AElementwiseOp            = typename decltype(input_op)::Op;
    using BElementwiseOp            = typename decltype(weight_op)::Op;
    using CDEElementwiseOp          = typename decltype(output_op)::Op;
};

} // namespace ck_tile::builder::factory::internal
