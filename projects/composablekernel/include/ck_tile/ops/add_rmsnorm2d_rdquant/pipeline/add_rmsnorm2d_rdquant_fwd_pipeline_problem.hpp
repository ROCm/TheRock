// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

// X = A + B, Y = Rmsnorm2d(X), QY = RowwiseDynamicQuant(Y) = SaturateCast(Y / YScale)
template <typename ADataType_,
          typename BDataType_,
          typename GammaDataType_,
          typename ComputeDataType_,
          typename XDataType_,
          typename YScaleDataType_,
          typename QYDataType_,
          typename BlockShape_,
          bool kPadN_,
          bool kSaveX_,
          bool kThreePass_>
struct AddRmsnorm2dRdquantFwdPipelineProblem
{
    using ADataType       = remove_cvref_t<ADataType_>;
    using BDataType       = remove_cvref_t<BDataType_>;
    using GammaDataType   = remove_cvref_t<GammaDataType_>;
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;
    using XDataType       = remove_cvref_t<XDataType_>;
    using YScaleDataType  = remove_cvref_t<YScaleDataType_>;
    using QYDataType      = remove_cvref_t<QYDataType_>;
    using BlockShape      = remove_cvref_t<BlockShape_>;

    static constexpr bool kNeedCrossLaneSync = BlockShape::ThreadPerWarp_N > 1;
    static constexpr bool kNeedCrossWarpSync = BlockShape::WarpPerBlock_N > 1;

    static constexpr bool kPadN      = kPadN_;
    static constexpr bool kSaveX     = kSaveX_;
    static constexpr bool kThreePass = kThreePass_;
};

} // namespace ck_tile
