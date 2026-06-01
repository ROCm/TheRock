// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename QDataType_,
          typename KDataType_,
          typename VDataType_,
          typename GemmDataType_,
          typename LSEDataType_,
          typename AccDataType_,
          typename DDataType_,
          typename BiasDataType_,
          typename RandValOutputDataType_,
          typename ODataType_,
          typename OGradDataType_,
          typename QGradDataType_,
          typename KGradDataType_,
          typename VGradDataType_,
          typename BiasGradDataType_,
          typename BlockFmhaShape_,
          bool kIsGroupMode_,
          bool kIsDeterministic_,
          typename FmhaMask_,
          typename FmhaDropout_,
          bool kUseTrLoad_,
          typename Traits_>
struct BlockFmhaBwdPipelineProblem
{
    using QDataType             = remove_cvref_t<QDataType_>;
    using KDataType             = remove_cvref_t<KDataType_>;
    using VDataType             = remove_cvref_t<VDataType_>;
    using GemmDataType          = remove_cvref_t<GemmDataType_>;
    using LSEDataType           = remove_cvref_t<LSEDataType_>;
    using AccDataType           = remove_cvref_t<AccDataType_>;
    using DDataType             = remove_cvref_t<DDataType_>;
    using BiasDataType          = remove_cvref_t<BiasDataType_>;
    using RandValOutputDataType = remove_cvref_t<RandValOutputDataType_>;
    using ODataType             = remove_cvref_t<ODataType_>;
    using OGradDataType         = remove_cvref_t<OGradDataType_>;
    using QGradDataType         = remove_cvref_t<QGradDataType_>;
    using KGradDataType         = remove_cvref_t<KGradDataType_>;
    using VGradDataType         = remove_cvref_t<VGradDataType_>;
    using BiasGradDataType      = remove_cvref_t<BiasGradDataType_>;
    using BlockFmhaShape        = remove_cvref_t<BlockFmhaShape_>;
    using FmhaMask              = remove_cvref_t<FmhaMask_>;
    using FmhaDropout           = remove_cvref_t<FmhaDropout_>;
    using Traits                = remove_cvref_t<Traits_>;

    static constexpr index_t kBlockSize    = BlockFmhaShape::NumWarps * get_warp_size();
    static constexpr bool kIsGroupMode     = kIsGroupMode_;
    static constexpr bool kIsDeterministic = kIsDeterministic_;
    static constexpr bool kUseTrLoad       = kUseTrLoad_;

    // attributes from traits
    static constexpr index_t kPadHeadDimQ = Traits::kPadHeadDimQ;
    static constexpr index_t kPadHeadDimV = Traits::kPadHeadDimV;
    static constexpr auto BiasEnum        = Traits::BiasEnum;
    static constexpr bool kHasBiasGrad    = Traits::kHasBiasGrad;
    static constexpr index_t kBlockPerCu  = Traits::kBlockPerCu;
};

template <typename ODataType_,
          typename OGradDataType_,
          typename DDataType_,
          index_t kBlockSize_,
          index_t kVHeaddim_,
          bool kIsGroupMode_,
          typename Traits_>
struct BlockFmhaBwdOGradDotOPipelineProblem
{
    using ODataType     = remove_cvref_t<ODataType_>;
    using OGradDataType = remove_cvref_t<OGradDataType_>;
    using DDataType     = remove_cvref_t<DDataType_>;
    using Traits        = remove_cvref_t<Traits_>;

    static_assert(0 < kBlockSize_ && kBlockSize_ % get_warp_size() == 0,
                  "kBlockSize should be divisible by get_warp_size()");

    static constexpr index_t kBlockSize = kBlockSize_;
    static constexpr index_t kVHeaddim  = kVHeaddim_;
    static constexpr bool kIsGroupMode  = kIsGroupMode_;

    // attributes from traits
    static constexpr bool kPadSeqLenQ    = Traits::kPadSeqLenQ;
    static constexpr bool kPadHeadDimV   = Traits::kPadHeadDimV;
    static constexpr index_t kBlockPerCu = Traits::kBlockPerCu;
};

template <typename AccDataType_,
          typename QGradDataType_,
          index_t kBlockSize_,
          index_t kM0_,
          index_t kN0_,
          index_t kQKHeaddim_,
          bool kIsGroupMode_,
          bool kIsDeterministic_,
          typename Traits_>
struct BlockFmhaBwdConvertQGradPipelineProblem
{
    using AccDataType   = remove_cvref_t<AccDataType_>;
    using QGradDataType = remove_cvref_t<QGradDataType_>;
    using Traits        = remove_cvref_t<Traits_>;

    static_assert(0 < kBlockSize_ && kBlockSize_ % get_warp_size() == 0,
                  "kBlockSize should be divisible by get_warp_size()");

    static constexpr index_t kBlockSize    = kBlockSize_;
    static constexpr index_t kM0           = kM0_;
    static constexpr index_t kN0           = kN0_;
    static constexpr index_t kQKHeaddim    = kQKHeaddim_;
    static constexpr bool kIsGroupMode     = kIsGroupMode_;
    static constexpr bool kIsDeterministic = kIsDeterministic_;

    // attributes from traits
    static constexpr bool kPadSeqLenQ    = Traits::kPadSeqLenQ;
    static constexpr bool kPadHeadDimQ   = Traits::kPadHeadDimQ;
    static constexpr index_t kBlockPerCu = Traits::kBlockPerCu;
};

} // namespace ck_tile
