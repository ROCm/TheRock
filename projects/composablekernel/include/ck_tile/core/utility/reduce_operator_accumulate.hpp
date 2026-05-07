// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"

namespace ck_tile {

/// @brief Accumulate with index tracking reductions, provides deterministic first occurring index
struct AccumulateWithIndex
{
    template <typename ReduceOp, typename T, typename IndexType>
    CK_TILE_HOST_DEVICE void operator()(const ReduceOp& reduce_func,
                                        T& current_value,
                                        IndexType& current_index,
                                        const T& new_value,
                                        const IndexType& new_index) const
    {
        bool changed  = false;
        current_value = reduce_func(current_value, new_value, changed);

        if(changed)
        {
            current_index = new_index;
        }
        else if(new_index < current_index)
        {
            bool reverse_changed = false;
            reduce_func(new_value, current_value, reverse_changed);

            if(!reverse_changed)
            {
                current_index = new_index;
            }
        }
    }
};

struct Accumulate
{
    template <typename ReduceOp, typename T>
    CK_TILE_HOST_DEVICE void
    operator()(const ReduceOp& reduce_func, T& current_value, const T& new_value) const
    {
        current_value = reduce_func(current_value, new_value);
    }
};

} // namespace ck_tile
