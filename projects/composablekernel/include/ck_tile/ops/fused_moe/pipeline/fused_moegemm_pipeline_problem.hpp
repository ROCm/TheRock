// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// TODO: alow 2 gemm have different type
template <typename ADataType_,
          typename GDataType_,
          typename DDataType_,
          typename AccDataType_,
          typename ODataType_,
          typename AScaleDataType_,
          typename GScaleDataType_,
          typename DScaleDataType_,
          typename YSmoothScaleDataType_,
          typename TopkWeightDataType_,
          typename IndexDataType_,  // data type for all indexing
          typename GateActivation_, // = ck_tile::element_wise::Silu,
          typename BlockShape_,     // shoule be FusedMoeGemmShape
          typename Traits_>
struct FusedMoeGemmPipelineProblem
{
    using ADataType            = remove_cvref_t<ADataType_>;
    using GDataType            = remove_cvref_t<GDataType_>;
    using DDataType            = remove_cvref_t<DDataType_>;
    using AccDataType          = remove_cvref_t<AccDataType_>;
    using ODataType            = remove_cvref_t<ODataType_>;
    using AScaleDataType       = remove_cvref_t<AScaleDataType_>;
    using GScaleDataType       = remove_cvref_t<GScaleDataType_>;
    using DScaleDataType       = remove_cvref_t<DScaleDataType_>;
    using YSmoothScaleDataType = remove_cvref_t<YSmoothScaleDataType_>;
    using TopkWeightDataType   = remove_cvref_t<TopkWeightDataType_>;
    using IndexDataType        = remove_cvref_t<IndexDataType_>;

    // the input for next gemm should have same time as
    using YDataType = ADataType;

    using GateActivation = remove_cvref_t<GateActivation_>;
    using BlockShape     = remove_cvref_t<BlockShape_>;
    using Traits         = remove_cvref_t<Traits_>;
};
} // namespace ck_tile
