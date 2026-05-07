// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// supports 2D transpose which will store to lds,
// then use ds_read_b*_tr_b* instruction to get the transposed data
template <typename DataType_,
          typename BlockTile, // sequence<block_x, block_y>
          typename NumWarps,
          bool kPadM_,
          bool kPadN_>
struct BatchedTransposeLdsProblem
{
    using DataType = remove_cvref_t<DataType_>;

    static constexpr index_t kRowWarps_    = NumWarps::at(number<0>{});
    static constexpr index_t kColWarps_    = NumWarps::at(number<1>{});
    static constexpr index_t kRowPerBlock_ = BlockTile::at(number<0>{});
    static constexpr index_t kColPerBlock_ = BlockTile::at(number<1>{});

    static constexpr index_t kBlockSize = get_warp_size() * kRowWarps_ * kColWarps_;
    // warps per block
    static constexpr index_t kLeadNumWarps   = kColWarps_;
    static constexpr index_t kSecondNumWarps = kRowWarps_;

    static constexpr index_t kLeadSizePerBlock   = kColPerBlock_;
    static constexpr index_t kSecondSizePerBlock = kRowPerBlock_;

    static constexpr index_t kQuadrantLeadDim   = LaneGroupTransposeTraits<DataType>::kleadDim;
    static constexpr index_t kQuadrantSecondDim = LaneGroupTransposeTraits<DataType>::ksecondDim;

    static_assert(kLeadSizePerBlock % kLeadNumWarps == 0,
                  "block dim should be divided by warp count!");
    static_assert(kSecondSizePerBlock % kSecondNumWarps == 0,
                  "block dim should be divided by warp count!");
    // rows/cols per warp
    static constexpr index_t kLeadSizePerWarp   = kLeadSizePerBlock / kLeadNumWarps;
    static constexpr index_t kSecondSizePerWarp = kSecondSizePerBlock / kSecondNumWarps;

    static_assert(kLeadSizePerWarp % kQuadrantLeadDim == 0,
                  "xdl dim should be divided by quad dim!");
    static_assert(kSecondSizePerWarp % kQuadrantSecondDim == 0,
                  "xdl dim should be divided by quad dim!");
    // xdl rows/cols is divided into quadrants.
    static constexpr index_t kQuadNumPerLeadDim   = kLeadSizePerWarp / kQuadrantLeadDim;
    static constexpr index_t kQuadNumPerSecondDim = kSecondSizePerWarp / kQuadrantSecondDim;

    static constexpr index_t kIterationsInSecondDim =
        kQuadNumPerLeadDim * kQuadNumPerSecondDim * 16 / get_warp_size();

    // definitions to adapt to BatchedTransposeKernel

    // FIXME: support padding
    static constexpr bool kPadM = kPadM_;
    static constexpr bool kPadN = kPadN_;

    static constexpr auto kMPerBlock = kSecondSizePerBlock;
    static constexpr auto kNPerBlock = kLeadSizePerBlock;

    // 128-bit is the max single-instruction bandwidth for load/store
    static constexpr index_t MaxLoadStoreSize = 16;
    static constexpr auto VectorSizeInput     = kPadN ? 1 : MaxLoadStoreSize / sizeof(DataType);
    static constexpr auto VectorSizeOutput    = kPadM ? 1 : MaxLoadStoreSize / sizeof(DataType);
    static constexpr auto LDSVectorSize       = MaxLoadStoreSize / sizeof(DataType);
};

} // namespace ck_tile
