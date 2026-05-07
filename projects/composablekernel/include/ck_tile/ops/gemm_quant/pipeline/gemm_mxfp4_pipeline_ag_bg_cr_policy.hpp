// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "gemm_group_quant_utils.hpp"

namespace ck_tile {

struct GemmMxFp4PipelineAgBgCrPolicy : public UniversalGemmPipelineAgBgCrPolicy
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
        constexpr index_t NPerBlockBQ = NPerBlock / Problem::QuantGroupSize::kN;
        constexpr index_t KPerBlock   = Problem::BlockGemmShape::kK;
        constexpr index_t KPerBlockBQ = KPerBlock / Problem::QuantGroupSize::kK;

        static_assert(std::is_same_v<BQLayout, ck_tile::tensor_layout::gemm::ColumnMajor>);
        return GetABQGlobalVectorLoadSize<Problem, BQDataType, NPerBlockBQ, KPerBlockBQ>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBRegTileDistribution()
    {
        using BLayout = remove_cvref_t<typename Problem::BLayout>;

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;
        constexpr index_t VecLoadSize =
            Problem::FixedVectorSize ? Problem::VectorSizeB : GetVectorSizeB<Problem>();
        constexpr index_t NumWaveGroups = Problem::NumWaveGroups;
        // Tile: KPerBlock X NPerBlock
        if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            using TileEncodingPattern =
                tile_distribution_encoding_pattern_2d<BlockSize,
                                                      KPerBlock,
                                                      NPerBlock,
                                                      VecLoadSize,
                                                      getBTileAccessPattern(),
                                                      NumWaveGroups>;
            return TileEncodingPattern::make_2d_static_tile_distribution();
        }
        // Tile: NPerBlock X KPerBlock
        else
        {
            using TileEncodingPattern =
                tile_distribution_encoding_pattern_2d<BlockSize,
                                                      NPerBlock,
                                                      KPerBlock,
                                                      VecLoadSize,
                                                      getBTileAccessPattern(),
                                                      NumWaveGroups>;
            return TileEncodingPattern::make_2d_static_tile_distribution();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBQDramTileDistribution()
    {
        // using BLayout = remove_cvref_t<typename Problem::BLayout>;

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t NPerBlock = Problem::BlockGemmShape::kN;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t KScale = KPerBlock / Problem::QuantGroupSize::kK; // k_scale num  //2
        constexpr index_t VecLoadSize =
            Problem::FixedVectorSize ? Problem::VectorSizeB : GetVectorSizeB<Problem>();
        constexpr index_t NumWaveGroups = Problem::NumWaveGroups;

        constexpr index_t warp_size  = get_warp_size();
        constexpr index_t num_warps  = BlockSize / get_warp_size();
        constexpr index_t LargestVec = (KPerBlock * NPerBlock) / (num_warps * warp_size);
        constexpr index_t b_vec      = VecLoadSize > LargestVec ? LargestVec : VecLoadSize;
        constexpr index_t K0         = KPerBlock / b_vec;
        constexpr index_t K1         = K0 / KScale;
        constexpr index_t K3         = K0 / K1;
        constexpr index_t K2         = 1;

        constexpr index_t N0 = num_warps / NumWaveGroups;
        constexpr index_t N1 = warp_size / K0;
        constexpr index_t N2 = NPerBlock / (N0 * N1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<K1>,
                                       tuple<sequence<N0, N1, N2>, sequence<K3, K2>>,
                                       tuple<sequence<1>, sequence<1, 2, 0>>,
                                       tuple<sequence<0>, sequence<1, 0, 0>>,
                                       sequence<1, 2>,
                                       sequence<2, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        using BlockWarps = typename Problem::BlockGemmShape::BlockWarps;
        using WarpTile   = typename Problem::BlockGemmShape::WarpTile;

        static_assert(Problem::QuantGroupSize::kK % WarpTile::at(I2) == 0,
                      "KPerWarpGemm must be a multiple of QuantGroupSize!");

        using WarpGemm = WarpGemmDispatcher<typename Problem::ComputeDataType,
                                            typename Problem::ComputeDataType,
                                            typename Problem::CDataType,
                                            WarpTile::at(I0),
                                            WarpTile::at(I1),
                                            WarpTile::at(I2),
                                            Problem::TransposeC>;
        static_assert(std::is_same_v<typename Problem::ComputeDataType, fp8_t> ||
                      std::is_same_v<typename Problem::ComputeDataType, bf8_t> ||
                      std::is_same_v<typename Problem::ComputeDataType, bf16_t>);
        static_assert(std::is_same_v<typename Problem::CDataType, float>);

        using BlockGemmPolicy = BlockGemmASmemBSmemCRegV1CustomPolicy<
            typename Problem::ADataType,
            std::conditional_t<std::is_same_v<typename Problem::BDataType, pk_fp4_raw_t>,
                               typename Problem::ADataType,
                               typename Problem::BDataType>,
            typename Problem::CDataType,
            BlockWarps,
            WarpGemm>;

        return BlockUniversalGemmAsBsCr<Problem, BlockGemmPolicy>{};
    }
};

} // namespace ck_tile
