// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "gemm_group_quant_utils.hpp"

namespace ck_tile {

struct GemmBQuantPipelineAgBgCrDefaultPolicy : public UniversalGemmPipelineAgBgCrPolicy
{
    using Base = UniversalGemmPipelineAgBgCrPolicy;
    using Base::I0;
    using Base::I1;
    using Base::I2;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeBQ()
    {
        using BQLayout                = remove_cvref_t<typename Problem::BQLayout>;
        using BQDataType              = remove_cvref_t<typename Problem::BQDataType>;
        constexpr index_t NPerBlock   = Problem::BlockGemmShape::kN;
        constexpr index_t NPerBlockBQ = NPerBlock / Problem::BQuantGroupSize::kN;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;
        constexpr index_t KPerBlockBQ = KPerBlock / Problem::BQuantGroupSize::kK;

        // Support both RowMajor and ColumnMajor layouts for BQ
        if constexpr(std::is_same_v<BQLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            return GetABQGlobalVectorLoadSize<Problem, BQDataType, KPerBlockBQ, NPerBlockBQ>();
        }
        else
        {
            static_assert(std::is_same_v<BQLayout, ck_tile::tensor_layout::gemm::ColumnMajor>);
            return GetABQGlobalVectorLoadSize<Problem, BQDataType, NPerBlockBQ, KPerBlockBQ>();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBQDramTileDistribution()
    {
        using BQLayout       = remove_cvref_t<typename Problem::BQLayout>;
        using BlockGemmShape = typename Problem::BlockGemmShape;

        constexpr index_t BlockSize    = Problem::kBlockSize;
        constexpr index_t NPerBlock    = Problem::BlockGemmShape::kN;
        constexpr index_t NPerBlockBQ  = NPerBlock / Problem::BQuantGroupSize::kN;
        constexpr index_t KPerBlock    = Problem::BlockGemmShape::kK;
        constexpr index_t KPerBlockBQ  = KPerBlock / Problem::BQuantGroupSize::kK;
        constexpr bool PreshuffleQuant = Problem::Traits::PreshuffleQuant;

        using WarpTile = typename Problem::BlockGemmShape::WarpTile;
        using WarpGemm = WarpGemmDispatcher<typename Problem::ComputeDataType,
                                            typename Problem::ComputeDataType,
                                            typename Problem::CDataType,
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC>;

        if constexpr(PreshuffleQuant)
        {
            using TileEncodingPattern = tile_distribution_encoding_pattern_bq<
                BlockGemmShape,
                WarpGemm,
                BlockSize,
                NPerBlock / WarpGemm::kN,
                ck_tile::integer_least_multiple(WarpGemm::kN * KPerBlockBQ, get_warp_size()),
                Problem::BQuantGroupSize::kN,
                Problem::BQuantGroupSize::kK,
                BQLayout,
                PreshuffleQuant>;
            return TileEncodingPattern::make_2d_static_tile_distribution();
        }
        else
        {
            // KPerTile and NPerTile are LOGICAL dimensions (K quant groups and N quant groups)
            using TileEncodingPattern =
                tile_distribution_encoding_pattern_bq<BlockGemmShape,
                                                      WarpGemm,
                                                      BlockSize,
                                                      KPerBlockBQ, // Logical K dimension
                                                      NPerBlockBQ, // Logical N dimension
                                                      Problem::BQuantGroupSize::kN,
                                                      Problem::BQuantGroupSize::kK,
                                                      BQLayout>;

            return TileEncodingPattern::make_2d_static_tile_distribution();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        static_assert(Problem::BQuantGroupSize::kK % WarpTile::at(I2) == 0,
                      "KPerWarpGemm must be a multiple of QuantGroupSize!");

        using WarpGemm = WarpGemmDispatcher<typename Problem::ComputeDataType,
                                            typename Problem::ComputeDataType,
                                            typename Problem::CDataType,
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC>;
        static_assert(std::is_same_v<typename Problem::ComputeDataType, fp8_t> ||
                      std::is_same_v<typename Problem::ComputeDataType, bf8_t>);
        static_assert(std::is_same_v<typename Problem::CDataType, float>);
        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::ADataType,
                                                                      typename Problem::BDataType,
                                                                      typename Problem::CDataType,
                                                                      BlockWarps,
                                                                      WarpGemm>;
        return BQuantBlockUniversalGemmAsBsCr<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
