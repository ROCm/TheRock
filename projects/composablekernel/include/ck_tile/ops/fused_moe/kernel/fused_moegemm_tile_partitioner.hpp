// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile {

template <typename BlockShape_>
struct FusedMoeGemmTilePartitioner_Linear
{
    //  FusedMoeGemmShape
    using BlockShape = ck_tile::remove_cvref_t<BlockShape_>;

    static constexpr const char* name = "lin";

    CK_TILE_DEVICE auto operator()(ck_tile::index_t /*num_sorted_tiles*/,
                                   ck_tile::index_t /*intermediate_size*/)
    {
        index_t i_n = blockIdx.x;
        index_t i_m = blockIdx.y;

        return ck_tile::make_tuple(i_m, i_n);
    }

    CK_TILE_HOST static constexpr auto GridSize(index_t max_tokens, index_t intermediate_size)
    {
        // TODO: this may need tuning
        index_t ms = ck_tile::integer_divide_ceil(max_tokens, BlockShape::Block_M0);
        index_t ns = ck_tile::integer_divide_ceil(intermediate_size, BlockShape::Block_N0);
        return dim3(ns, ms, 1);
    }
};
} // namespace ck_tile
