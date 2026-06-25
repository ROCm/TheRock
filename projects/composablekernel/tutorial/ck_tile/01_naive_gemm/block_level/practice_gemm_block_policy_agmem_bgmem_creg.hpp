// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/host.hpp"
#include "ck_tile/core.hpp"

#include "../warp_level/practice_gemm_warp_policy_asmem_bsmem_creg.hpp"
#include "../warp_level/practice_gemm_warp_pipeline_asmem_bsmem_creg.hpp"

namespace ck_tile {

template <typename ADataType_,
          typename BDataType_,
          typename CDataType_,
          typename AccDataType_,
          typename Shape_>
struct PracticeGemmBlockPipelineProblem
{
    using ADataType   = ADataType_;
    using BDataType   = BDataType_;
    using CDataType   = CDataType_;
    using AccDataType = AccDataType_;
    using Shape       = Shape_;
};

struct PracticeGemmBlockPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetPracticeWaveGemmPipeline()
    {
        return PracticeGemmWarpPipelineASmemBSmemCreg<Problem>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::Shape::BlockTile::at(number<0>{});
        constexpr index_t kKPerBlock = Problem::Shape::BlockTile::at(number<2>{});
        constexpr index_t kKPack     = 8;

        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock / kKPack>{}, number<kKPack>{}),
            make_tuple(number<kKPerBlock>{}, number<kKPack>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        return a_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::Shape::BlockTile::at(number<1>{});
        constexpr index_t kKPerBlock = Problem::Shape::BlockTile::at(number<2>{});
        constexpr index_t kKPack     = 8;

        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock>{}, number<kKPerBlock / kKPack>{}, number<kKPack>{}),
            make_tuple(number<kKPerBlock>{}, number<kKPack>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kNPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(sequence<0>{}, sequence<1, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return b_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        using ADataType          = remove_cvref_t<typename Problem::ADataType>;
        using BlockGemm          = remove_cvref_t<decltype(GetPracticeWaveGemmPipeline<Problem>())>;
        constexpr index_t kMWarp = BlockGemm::MWarp;
        constexpr index_t kNWarp = BlockGemm::NWarp;
        constexpr index_t kBlockSize = kMWarp * kNWarp * get_warp_size();

        constexpr index_t kMPerBlock = Problem::Shape::BlockTile::at(number<0>{});
        constexpr index_t kKPerBlock = Problem::Shape::BlockTile::at(number<2>{});

        constexpr index_t K1 = 16 / sizeof(ADataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t M2 = get_warp_size() / K0;
        // coalesce reading for each blocks
        constexpr index_t M1 = kBlockSize / get_warp_size();
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,                                   // replication
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1>>, // hierarchy
                                       tuple<sequence<1>, sequence<1, 2>>,            // parallelism
                                       tuple<sequence<1>, sequence<2, 0>>,            // paralleism
                                       sequence<1, 2>,                                // yield
                                       sequence<0, 1>>{});                            // yield
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        using BDataType          = remove_cvref_t<typename Problem::BDataType>;
        using BlockGemm          = remove_cvref_t<decltype(GetPracticeWaveGemmPipeline<Problem>())>;
        constexpr index_t kMWarp = BlockGemm::MWarp;
        constexpr index_t kNWarp = BlockGemm::NWarp;
        constexpr index_t kBlockSize = kMWarp * kNWarp * get_warp_size();

        constexpr index_t kNPerBlock = Problem::Shape::BlockTile::at(number<1>{});
        constexpr index_t kKPerBlock = Problem::Shape::BlockTile::at(number<2>{});

        constexpr index_t K1 = 16 / sizeof(BDataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;
        // coalesce reading for each blocks
        constexpr index_t N1 = kBlockSize / get_warp_size();
        constexpr index_t N0 = kNPerBlock / (N2 * N1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }
};

} // namespace ck_tile
