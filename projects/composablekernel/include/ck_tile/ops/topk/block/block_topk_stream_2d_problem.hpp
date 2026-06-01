// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

/*
simple 2d topk implementation, along row (dim=1)
requirement:
    1). each row is within a warp
*/
template <typename DataType_, typename IndexType_, index_t ColLanes_>
struct BlockTopkStream2DProblem
{
    using DataType                    = remove_cvref_t<DataType_>;
    using IndexType                   = remove_cvref_t<IndexType_>;
    static constexpr index_t ColLanes = ColLanes_;
};
} // namespace ck_tile
