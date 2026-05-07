// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"

namespace ck_tile {

// This pipeline is qkv all located in LDS
struct BlockFmhaPipelineQSKSVSDefaultPolicy
    : BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ false,
                                          /* AsyncCopy = */ false,
                                          /* NumPrefetchK = */ 1,
                                          /* NumPrefetchV = */ 1>
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeK()
    {
        return MakeKLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::KDataType);
    } // namespace ck_tile

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeV()
    {
        return MakeVLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::VDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return max(GetSmemSizeQ<Problem>() + GetSmemSizeK<Problem>(), GetSmemSizeV<Problem>()) +
               GetSmemSizeDropout<Problem>();
    }
};

} // namespace ck_tile
