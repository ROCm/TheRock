// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <index_t kM0_, index_t kN0_, index_t kK0_, index_t kN1_>
struct FmhaFwdAppendKVTilePartitioner
{
    static constexpr ck_tile::index_t kM0 = kM0_;
    static constexpr ck_tile::index_t kN0 = kN0_;
    static constexpr ck_tile::index_t kK0 = kK0_;
    static constexpr ck_tile::index_t kN1 = kN1_;

    static_assert(kK0 == kN1);

    CK_TILE_HOST static constexpr auto GridSize(ck_tile::index_t batch_size,
                                                ck_tile::index_t nhead,
                                                ck_tile::index_t seqlen_q,
                                                ck_tile::index_t seqlen_knew)
    {
        // TODO: this may need tuning
        return dim3(std::max(ck_tile::integer_divide_ceil(seqlen_q, kM0),
                             ck_tile::integer_divide_ceil(seqlen_knew, kN0)),
                    nhead,
                    batch_size);
    }

    CK_TILE_DEVICE auto operator()()
    {
        const index_t i_tile  = blockIdx.x;
        const index_t i_nhead = blockIdx.y;
        const index_t i_batch = blockIdx.z;

        return ck_tile::make_tuple(i_tile, i_nhead, i_batch);
    }
};

} // namespace ck_tile
