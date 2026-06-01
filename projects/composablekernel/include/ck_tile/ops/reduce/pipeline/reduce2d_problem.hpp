// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename XDataType_,
          typename ComputeDataType_,
          typename YDataType_,
          typename BlockShape_,
          typename ReduceOp_,
          typename KeptDim_,
          typename ReduceDims_,
          index_t Rank_,
          bool OutputIndex_ = false>
struct Reduce2dProblem
{
    using XDataType       = remove_cvref_t<XDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using YDataType       = remove_cvref_t<YDataType_>;
    using BlockShape      = remove_cvref_t<BlockShape_>;
    using ReduceOp        = ReduceOp_;
    using KeptDim         = remove_cvref_t<KeptDim_>;
    using ReduceDims      = remove_cvref_t<ReduceDims_>;

    static constexpr index_t Rank            = Rank_;
    static constexpr index_t NumReduceDim    = ReduceDims::size();
    static constexpr bool kOutputIndex       = OutputIndex_;
    static constexpr bool kNeedCrossLaneSync = BlockShape::ThreadPerWarp_N > 1;
    static constexpr bool kNeedCrossWarpSync = BlockShape::WarpPerBlock_N > 1;
};

} // namespace ck_tile
