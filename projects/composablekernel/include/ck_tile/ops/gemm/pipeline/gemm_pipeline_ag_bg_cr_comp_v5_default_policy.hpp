// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"

namespace ck_tile {
// Default policy for GemmPipelineAGmemBGmemCregComputeV5, except the block gemm method, it shares
// the same vector size implementation, SmemSize, Global memory tile distiribution as the
// UniversalGemm Pipeline Policy.
// Default policy class should not be templated, put template on
// member functions instead.
struct GemmPipelineAgBgCrCompV5DefaultPolicy
    : public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompV5DefaultPolicy>
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        // using AccDataType     = float;
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm   = WarpGemmDispatcher<typename Problem::ADataType,
                                              typename Problem::BDataType,
                                              typename Problem::CDataType, // AccDataType
                                              WarpTile::at(I0),
                                              WarpTile::at(I1),
                                              WarpTile::at(I2),
                                              Problem::TransposeC>;

        using BlockGemmPolicy = BlockGemmARegBRegCRegV1CustomPolicy<typename Problem::ADataType,
                                                                    typename Problem::BDataType,
                                                                    typename Problem::CDataType,
                                                                    BlockWarps,
                                                                    WarpGemm>;

        return BlockGemmARegBRegCRegV1<Problem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr index_t GetSmemSizeC()
    {
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;

        return integer_least_multiple(sizeof(typename Problem::CDataType) * MPerBlock * NPerBlock,
                                      16);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        constexpr index_t smem_size_a = GetSmemSizeA<Problem>();
        constexpr index_t smem_size_b = GetSmemSizeB<Problem>();
        constexpr index_t smem_size_c = GetSmemSizeC<Problem>();

        return smem_size_a + smem_size_b >= smem_size_c ? (smem_size_a + smem_size_b)
                                                        : (smem_size_c);
    }
};
} // namespace ck_tile
