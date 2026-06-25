// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/elementwise.hpp"
// Overload methods required for the parametrize tests

// Overload for PassThrough (no parameter)
inline ck_tile::element_wise::PassThrough make_elementwise_op(int32_t,
                                                              ck_tile::element_wise::PassThrough)
{
    return ck_tile::element_wise::PassThrough{};
}

// Overload for UnaryDivide (needs parameter)
inline ck_tile::element_wise::UnaryDivide make_elementwise_op(int32_t total_reduce_elements,
                                                              ck_tile::element_wise::UnaryDivide)
{
    return ck_tile::element_wise::UnaryDivide{total_reduce_elements};
}

// Overload for UnarySquare (no parameter)
inline ck_tile::element_wise::UnarySquare make_elementwise_op(int32_t,
                                                              ck_tile::element_wise::UnarySquare)
{
    return ck_tile::element_wise::UnarySquare{};
}

template <typename... Ops>
auto make_elementwise_ops_tuple(int32_t total_reduce_elements, ck_tile::tuple<Ops...>)
{
    return ck_tile::make_tuple(make_elementwise_op(total_reduce_elements, Ops{})...);
}
