// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <typename XDataType_,
          typename XBiasDataType_,
          typename GammaDataType_,
          typename BetaDataType_,
          typename ComputeDataType_,
          typename YDataType_,
          typename MeanDataType_,
          typename InvStdDataType_,
          typename SmoothScaleDataType_,
          typename YScaleDataType_,
          typename BlockShape_,
          typename Traits_>
struct Layernorm2dFwdPipelineProblem
{
    using XDataType           = remove_cvref_t<XDataType_>;
    using XBiasDataType       = remove_cvref_t<XBiasDataType_>;
    using GammaDataType       = remove_cvref_t<GammaDataType_>;
    using BetaDataType        = remove_cvref_t<BetaDataType_>;
    using ComputeDataType     = remove_cvref_t<ComputeDataType_>;
    using YDataType           = remove_cvref_t<YDataType_>;
    using MeanDataType        = remove_cvref_t<MeanDataType_>;
    using InvStdDataType      = remove_cvref_t<InvStdDataType_>;
    using SmoothScaleDataType = remove_cvref_t<SmoothScaleDataType_>;
    using YScaleDataType      = remove_cvref_t<YScaleDataType_>;
    using BlockShape          = remove_cvref_t<BlockShape_>;

    static constexpr bool kNeedCrossLaneSync = BlockShape::ThreadPerWarp_N > 1;
    static constexpr bool kNeedCrossWarpSync = BlockShape::WarpPerBlock_N > 1;

    using Traits = remove_cvref_t<Traits_>;
};

} // namespace ck_tile
