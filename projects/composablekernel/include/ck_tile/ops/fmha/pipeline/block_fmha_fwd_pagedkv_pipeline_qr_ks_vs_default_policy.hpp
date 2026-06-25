// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2r1.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"

namespace ck_tile {

// This pipeline is qkv all located in LDS
struct BlockFmhaFwdPagedKVPipelineQRKSVSDefaultPolicy
    : BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                          /* AsyncCopy = */ false,
                                          /* NumPrefetchK = */ 1,
                                          /* NumPrefetchV = */ 1>
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::QDataType,
                             typename Problem::KDataType,
                             typename Problem::SaccDataType,
                             Problem::kNumGemm0Warps * get_warp_size(),
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kN0,
                                                    Problem::BlockFmhaShape::kK0>,
                                           typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm0WarpTile>>;

        constexpr auto warp_gemm = []() {
            if constexpr(get_warp_size() == 64 &&
                         std::is_same_v<typename Problem::QDataType, fp8_t> &&
                         std::is_same_v<typename Problem::KDataType, fp8_t> &&
                         std::is_same_v<typename Problem::SaccDataType, float>)
            {
                static_assert(Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}) == 32);
                static_assert(Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{}) == 32);
                static_assert(Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{}) == 32);

                // TODO: hard coded here. Otherwise, it produces incorrect results
                constexpr index_t swizzle_factor = 4;
                return WarpGemmMfmaFp8Fp8F32M32N32K32SwizzleBTransposedCDistribution<
                    swizzle_factor>{};
            }
            else
            {
                constexpr bool SwizzleA =
                    Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}) == 32;
                return WarpGemmDispatcher<typename Problem::QDataType,
                                          typename Problem::KDataType,
                                          typename Problem::SaccDataType,
                                          Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}),
                                          Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{}),
                                          Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{}),
                                          true, // TransposeC
                                          SwizzleA>{};
            }
        }();

        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV2CustomPolicy<typename Problem::QDataType,
                                                 typename Problem::KDataType,
                                                 typename Problem::SaccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                 decltype(warp_gemm)>;

        if constexpr(1 < Problem::kNumGemm0Warps)
        {
            if constexpr(128 >= Problem::BlockFmhaShape::kK0)
                return BlockGemmARegBSmemCRegV2R1<GemmProblem, BlockGemmPolicy>{};
            else
                return BlockGemmARegBSmemCRegV2<GemmProblem, BlockGemmPolicy>{};
        }
        else
            return BlockGemmARegBSmemCRegOneWarpV1<GemmProblem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
