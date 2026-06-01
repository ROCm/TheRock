// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename InDataType_,
          typename OutDataType_,
          typename ComputeDataType_,
          typename IndexDataType_,
          typename ReduceOp_,
          bool OutputIndex_,
          bool PropagateNan_,
          typename BlockShape_>
struct PoolProblem
{
    using InDataType      = remove_cvref_t<InDataType_>;
    using OutDataType     = remove_cvref_t<OutDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using IndexDataType   = remove_cvref_t<IndexDataType_>;
    using BlockShape      = remove_cvref_t<BlockShape_>;
    using ReduceOp        = ReduceOp_;
    using OutputIndex     = bool_constant<OutputIndex_>;
    using PropagateNan    = bool_constant<PropagateNan_>;

    static constexpr bool kOutputIndex       = OutputIndex_;
    static constexpr bool kPropagateNan      = PropagateNan_;
    static constexpr bool kNeedCrossLaneSync = BlockShape::ThreadPerWarp_N > 1;
    static constexpr bool kNeedCrossWarpSync = BlockShape::WarpPerBlock_N > 1;
};

} // namespace ck_tile
