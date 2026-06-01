// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"

namespace ck_tile {

struct BlockFmhaFwdSplitKVCombinePipelineDefaultPolicy
{
    template <index_t NumWarps, index_t M, index_t N, typename DataType>
    CK_TILE_HOST_DEVICE static constexpr auto GetMaxNumWarpsForTile()
    {
        static_assert(NumWarps == 1 || NumWarps == 2 || NumWarps == 4);

        constexpr index_t ElemPerThread = (M * N) / (NumWarps * get_warp_size());
        if constexpr(0 < ElemPerThread)
        {
            return NumWarps;
        }
        else
        { // try dividing tile by smaller # of warps
            return GetMaxNumWarpsForTile<NumWarps / 2, M, N, DataType>();
        }
    }

    template <index_t NumWarps, index_t M, index_t N, typename DataType>
    CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeForTile()
    {
        constexpr index_t MaxNumWarps = GetMaxNumWarpsForTile<NumWarps, M, N, DataType>();

        constexpr index_t ElemPerThread = (M * N) / (MaxNumWarps * get_warp_size());

        constexpr index_t MaxNPerThread = 16 / sizeof(DataType);
        return min(MaxNPerThread, ElemPerThread);
    }

    // alignment for dram lse tile (shape=[kMaxSplits, kM0])
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentLSE()
    {
        return GetVectorSizeForTile<Problem::kNumWarps,
                                    Problem::kMaxSplits,
                                    Problem::kM0,
                                    typename Problem::LSEDataType>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentOacc()
    {
        using OaccDataType = remove_cvref_t<typename Problem::OaccDataType>;

        constexpr index_t kNumWarps  = Problem::kNumWarps;
        constexpr index_t kMPerBlock = Problem::kM0;
        constexpr index_t kNPerBlock = Problem::kN1;

        constexpr index_t M1 = kNumWarps;
        constexpr index_t M2 = min(kMPerBlock / M1, get_warp_size());
        constexpr index_t N0 = get_warp_size() / M2;
        constexpr index_t N1 = kNPerBlock / N0;

        return min(N1, static_cast<index_t>(16 / sizeof(OaccDataType)));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentO()
    {
        return GetAlignmentOacc<Problem>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeLSEacc()
    {
        return sizeof(typename Problem::LSEDataType) *
               MakeLSEaccLdsBlockDescriptor<Problem>().get_element_space_size();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeOacc()
    {
        return sizeof(typename Problem::OaccDataType) *
               MakeOaccLdsBlockDescriptor<Problem>().get_element_space_size();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return GetSmemSizeLSEacc<Problem>() + GetSmemSizeOacc<Problem>();
    }

    // shape=[kMaxSplits, kM0]
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLSEaccDramTileDistribution()
    {
        using LSEDataType = remove_cvref_t<typename Problem::LSEDataType>;

        constexpr index_t kMPerBlock = Problem::kMaxSplits;
        constexpr index_t kNPerBlock = Problem::kM0;

        constexpr index_t MaxNumWarps =
            GetMaxNumWarpsForTile<Problem::kNumWarps, kNPerBlock, kMPerBlock, LSEDataType>();
        constexpr index_t Replicate = Problem::kNumWarps / MaxNumWarps;

        constexpr index_t NPerThread =
            GetVectorSizeForTile<MaxNumWarps, kMPerBlock, kNPerBlock, LSEDataType>();
        constexpr index_t NThreads = kNPerBlock / NPerThread;

        constexpr index_t MThreadsPerWarp = get_warp_size() / NThreads;
        constexpr index_t MPerThread      = kMPerBlock / (MaxNumWarps * MThreadsPerWarp);

        static_assert(MPerThread * MaxNumWarps * MThreadsPerWarp == kMPerBlock);
        static_assert(NThreads * NPerThread == kNPerBlock);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<Replicate>,
                                       tuple<sequence<MPerThread, MaxNumWarps, MThreadsPerWarp>,
                                             sequence<NThreads, NPerThread>>,
                                       tuple<sequence<0, 1>, sequence<1, 2>>,
                                       tuple<sequence<0, 1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    // 3d + padding, shape=[kMaxSplits, kM0]
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLSEaccLdsStoreBlockDescriptor()
    {
        using LSEDataType = remove_cvref_t<typename Problem::LSEDataType>;

        constexpr index_t kMPerBlock = Problem::kMaxSplits;
        constexpr index_t kNPerBlock = Problem::kM0;
        constexpr index_t NPack =
            GetVectorSizeForTile<Problem::kNumWarps, kMPerBlock, kNPerBlock, LSEDataType>();

        constexpr auto lse_acc_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock / NPack>{}, number<kMPerBlock>{}, number<NPack>{}),
            make_tuple(number<(kMPerBlock + 1) * NPack>{}, number<NPack>{}, number<1>{}),
            number<NPack>{},
            number<1>{});

        constexpr auto lse_acc_lds_block_desc = transform_tensor_descriptor(
            lse_acc_lds_block_desc_0,
            make_tuple(
                make_pass_through_transform(number<kMPerBlock>{}),
                make_merge_transform(make_tuple(number<kNPerBlock / NPack>{}, number<NPack>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return lse_acc_lds_block_desc;
    }

    // 3d + padding, shape=[kM0, kMaxSplits]
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLSEaccLdsBlockDescriptor()
    {
        using LSEDataType = remove_cvref_t<typename Problem::LSEDataType>;

        constexpr index_t kMPerBlock = Problem::kMaxSplits;
        constexpr index_t kNPerBlock = Problem::kM0;
        constexpr index_t NPack =
            GetVectorSizeForTile<Problem::kNumWarps, kMPerBlock, kNPerBlock, LSEDataType>();

        constexpr auto lse_acc_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock / NPack>{}, number<kMPerBlock>{}, number<NPack>{}),
            make_tuple(number<(kMPerBlock + 1) * NPack>{}, number<NPack>{}, number<1>{}),
            number<NPack>{},
            number<1>{});

        constexpr auto lse_acc_t_lds_block_desc = transform_tensor_descriptor(
            lse_acc_lds_block_desc_0,
            make_tuple(
                make_pass_through_transform(number<kMPerBlock>{}),
                make_merge_transform(make_tuple(number<kNPerBlock / NPack>{}, number<NPack>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<1>{}, sequence<0>{}));

        return lse_acc_t_lds_block_desc;
    }

    // 3d + padding, shape=[kNumWarps * kM0, kN1]
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOaccLdsBlockDescriptor()
    {
        using LSEDataType = remove_cvref_t<typename Problem::LSEDataType>;

        constexpr index_t kNumWarps  = Problem::kNumWarps;
        constexpr index_t kMPerBlock = kNumWarps * Problem::kM0;
        constexpr index_t kNPerBlock = Problem::kN1;
        constexpr index_t NPack =
            GetVectorSizeForTile<Problem::kNumWarps, kMPerBlock, kNPerBlock, LSEDataType>();

        constexpr auto o_acc_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock / NPack>{}, number<kMPerBlock>{}, number<NPack>{}),
            make_tuple(number<(kMPerBlock + 1) * NPack>{}, number<NPack>{}, number<1>{}),
            number<NPack>{},
            number<1>{});

        constexpr auto o_acc_lds_block_desc = transform_tensor_descriptor(
            o_acc_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kNPerBlock / NPack, NPack))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return o_acc_lds_block_desc;
    }

    // shape=[kM0, kMaxSplits]
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLSEaccRegTileDistribution()
    {
        constexpr index_t kMPerBlock = Problem::kM0;
        constexpr index_t kNPerBlock = Problem::kMaxSplits;

        constexpr index_t MaxNThreads = 8;
        constexpr index_t NThreads    = min(kNPerBlock, MaxNThreads);
        constexpr index_t NPerThread  = kNPerBlock / NThreads;

        constexpr index_t MPerThread     = 1;
        constexpr index_t MThreads       = kMPerBlock / MPerThread;
        constexpr index_t MThreadPerWarp = get_warp_size() / NThreads;

        constexpr index_t MaxNumWarps = (MThreads * NThreads) / get_warp_size();
        constexpr index_t Replicate   = Problem::kNumWarps / MaxNumWarps;

        static_assert(MaxNumWarps * MThreadPerWarp * MPerThread == kMPerBlock);
        static_assert(NThreads * NPerThread == kNPerBlock);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<Replicate>,
                                       tuple<sequence<MaxNumWarps, MThreadPerWarp, MPerThread>,
                                             sequence<NThreads, NPerThread>>,
                                       tuple<sequence<0, 1>, sequence<2, 1>>,
                                       tuple<sequence<0, 0>, sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<2, 1>>{});
    }

    // similar to MakeOaccResultDramTileDistribution(), but duplicate same 1-warp encoding kNumWarps
    // times on M direction
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOaccDramTileDistribution()
    {
        constexpr index_t kNumWarps  = Problem::kNumWarps;
        constexpr index_t kMPerBlock = Problem::kM0; // real kMPerBlock we want is (kNumWarps * kM0)
        constexpr index_t kNPerBlock = Problem::kN1;
        static_assert(get_warp_size() <= kMPerBlock * kNPerBlock);

        constexpr index_t M1 = 1; // compose encoding base on 1 warp
        constexpr index_t M2 = min(kMPerBlock / M1, get_warp_size());
        constexpr index_t N0 = get_warp_size() / M2;
        constexpr index_t N1 = kNPerBlock / N0;
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<kNumWarps, M0, M1, M2>, sequence<N0, N1>>,
                                       tuple<sequence<1, 1>, sequence<1, 2>>,
                                       tuple<sequence<0, 2>, sequence<3, 0>>,
                                       sequence<1, 2>,
                                       sequence<1, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOaccResultDramTileDistribution()
    {
        constexpr index_t kNumWarps  = Problem::kNumWarps;
        constexpr index_t kMPerBlock = Problem::kM0;
        constexpr index_t kNPerBlock = Problem::kN1;
        static_assert(kNumWarps * get_warp_size() <= kMPerBlock * kNPerBlock);

        constexpr index_t M1 = kNumWarps;
        constexpr index_t M2 = min(kMPerBlock / M1, get_warp_size());
        constexpr index_t N0 = get_warp_size() / M2;
        constexpr index_t N1 = kNPerBlock / N0;
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<M0, M1, M2>, sequence<N0, N1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }
};

} // namespace ck_tile
