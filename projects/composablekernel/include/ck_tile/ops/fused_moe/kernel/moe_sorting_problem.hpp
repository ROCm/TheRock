// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename IndexType_,
          typename WeightType_,
          index_t InternalLoadUnroll_,
          index_t ExpertTile_ = 0>
struct MoeSortingProblem
{
    // TODO: this kernel only support warp per row
    using WeightType = remove_cvref_t<WeightType_>;
    using IndexType  = remove_cvref_t<IndexType_>;

    static constexpr index_t WarpSize      = get_warp_size();
    static constexpr index_t WarpsPerBlock = 1;
    static constexpr index_t InternalLoadUnroll =
        InternalLoadUnroll_;                           // TODO: need better design(like tile size)
    static constexpr index_t ExpertTile = ExpertTile_; // TODO: only used in store out
};

template <typename IndexType_,
          typename WeightType_,
          index_t SubTokenTile_,    // 1,2,4,8, or 0 in the future
          bool SubTokenOneShot_,    // if we only loop over once or not
          bool LocalExpertMasking_, // used in EP case
          bool LocalToken_,         // used in EP case
          bool SkipExpertsWithZeroTokens_ = true,
          index_t ExpertTile_             = 0>
struct MoeSortingProblemEx
{
    // TODO: this kernel only support warp per row
    using WeightType = remove_cvref_t<WeightType_>;
    using IndexType  = remove_cvref_t<IndexType_>;

    static constexpr index_t WarpSize               = get_warp_size();
    static constexpr index_t WarpsPerBlock          = 1;
    static constexpr index_t SubTokenTile           = SubTokenTile_;
    static constexpr bool SubTokenOneShot           = SubTokenOneShot_;
    static constexpr bool LocalExpertMasking        = LocalExpertMasking_;
    static constexpr bool LocalToken                = LocalToken_;
    static constexpr bool SkipExpertsWithZeroTokens = SkipExpertsWithZeroTokens_;
    static_assert(SubTokenTile == 1 || SubTokenTile == 2 || SubTokenTile == 4 || SubTokenTile == 8);
    static constexpr index_t ExpertTile = ExpertTile_; // TODO: only used in store out
};

template <typename IndexType_,
          typename WeightType_, // used for expert mesh in ws
          typename MeshType_,
          index_t SubTokenTile_,    // 1,2,4,8
          bool LocalExpertMasking_, // used in EP case
          bool LocalToken_,         // used in EP case
          bool SkipExpertsWithZeroTokens_ = true>
struct MoeSortingProblemMp
{
    // TODO: this kernel only support warp per row
    using WeightType = remove_cvref_t<WeightType_>;
    using MeshType   = remove_cvref_t<MeshType_>;
    using IndexType  = remove_cvref_t<IndexType_>;

    static constexpr index_t SubTokenTile           = SubTokenTile_;
    static constexpr bool LocalExpertMasking        = LocalExpertMasking_;
    static constexpr bool LocalToken                = LocalToken_;
    static constexpr bool SkipExpertsWithZeroTokens = SkipExpertsWithZeroTokens_;
    static_assert(SubTokenTile == 1 || SubTokenTile == 2 || SubTokenTile == 4 ||
                  SubTokenTile == 8 || SubTokenTile == 16);
};

template <bool LocalToken_, index_t BlockSize_ = 1024, index_t Occu_ = 1>
struct MoeSortingClearWorkspaceProblem
{
    static constexpr bool LocalToken   = LocalToken_;
    static constexpr index_t BlockSize = BlockSize_;
    static constexpr index_t Occu      = Occu_;
};

} // namespace ck_tile
