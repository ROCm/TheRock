// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
namespace ck_tile {

// This pipeline is qkv all located in LDS
struct BlockFmhaFwdAppendKVPipelineDefaultPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        using QDataType = remove_cvref_t<typename Problem::QDataType>;

        return 16 / sizeof(QDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentK()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;

        return 16 / sizeof(KDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentV()
    {
        using VLayout   = remove_cvref_t<typename Problem::VLayout>;
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t kBlockSize   = Problem::kBlockSize;
            constexpr index_t kNPerBlock   = Problem::kN0;
            constexpr index_t kKPerBlock   = Problem::kN1;
            constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;

            // TODO: not correct!
            if constexpr(total_pixels > 4)
                return 4;
            else
                return 2;
        }
        else
        {
            return 16 / sizeof(VDataType);
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQNumElemsPerRead()
    {
        using DataType = typename Problem::QDataType;

        if constexpr(Problem::RotaryEnum == RotaryEmbeddingEnum::HALF_ROTATED)
        {
            /// NOTICE: we might need to lower down this to support smaller rotary_dim
            return 16 / sizeof(DataType);
        }
        else
        {
            return 16 / sizeof(DataType);
        }
    }

    template <typename Problem>
    CK_TILE_DEVICE static auto GetQThreadRangeAlongK()
    {
        static_assert(Problem::RotaryEnum != RotaryEmbeddingEnum::NONE);

        if constexpr(Problem::RotaryEnum == RotaryEmbeddingEnum::INTERLEAVED)
        {
            constexpr index_t KPerThread = GetQNumElemsPerRead<Problem>();
            static_assert(Problem::kK0 % KPerThread == 0);
            constexpr index_t KThreadPerBlock = Problem::kK0 / KPerThread;
            index_t start_pos                 = (get_thread_id() % KThreadPerBlock) * KPerThread;

            return make_tuple(start_pos, start_pos + KPerThread);
        }
        else
        {
            constexpr index_t KPerThread = GetQNumElemsPerRead<Problem>();
            static_assert(Problem::kK0 % KPerThread == 0);
            constexpr index_t KThreadPerBlock = Problem::kK0 / KPerThread;
            index_t start_pos                 = (get_thread_id() % KThreadPerBlock) * KPerThread;

            return make_tuple(start_pos, start_pos + KPerThread);
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::kM0;
        constexpr index_t kKPerBlock = Problem::kK0;

        constexpr index_t KPerThread      = GetQNumElemsPerRead<Problem>();
        constexpr index_t KThreadPerBlock = kKPerBlock / KPerThread;
        constexpr index_t MThreadPerWarp  = get_warp_size() / KThreadPerBlock;
        constexpr index_t NumWarps        = kBlockSize / get_warp_size();
        constexpr index_t MPerThread      = kMPerBlock / (NumWarps * MThreadPerWarp);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<MPerThread, NumWarps, MThreadPerWarp>,
                                             sequence<KThreadPerBlock, KPerThread>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetKnewNumElemsPerRead()
    {
        using DataType = typename Problem::KDataType;

        if constexpr(Problem::RotaryEnum == RotaryEmbeddingEnum::HALF_ROTATED)
        {
            /// NOTICE: we might need to lower down this to support smaller rotary_dim
            return 16 / sizeof(DataType);
        }
        else
        {
            return 16 / sizeof(DataType);
        }
    }

    template <typename Problem>
    CK_TILE_DEVICE static auto GetKnewThreadRangeAlongK()
    {
        static_assert(Problem::RotaryEnum != RotaryEmbeddingEnum::NONE);

        if constexpr(Problem::RotaryEnum == RotaryEmbeddingEnum::INTERLEAVED)
        {
            constexpr index_t KPerThread      = GetKnewNumElemsPerRead<Problem>();
            constexpr index_t KThreadPerBlock = Problem::kK0 / KPerThread;
            index_t start_pos                 = (get_thread_id() % KThreadPerBlock) * KPerThread;

            return make_tuple(start_pos, start_pos + KPerThread);
        }
        else
        {
            constexpr index_t KPerThread      = GetKnewNumElemsPerRead<Problem>();
            constexpr index_t KThreadPerBlock = Problem::kK0 / KPerThread;
            index_t start_pos                 = (get_thread_id() % KThreadPerBlock) * KPerThread;

            return make_tuple(start_pos, start_pos + KPerThread);
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKnewDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::kN0;
        constexpr index_t kKPerBlock = Problem::kK0;

        constexpr index_t KPerThread      = GetKnewNumElemsPerRead<Problem>();
        constexpr index_t KThreadPerBlock = kKPerBlock / KPerThread;
        constexpr index_t NThreadPerWarp  = get_warp_size() / KThreadPerBlock;
        constexpr index_t NumWarps        = kBlockSize / get_warp_size();
        constexpr index_t NPerThread      = kNPerBlock / (NumWarps * NThreadPerWarp);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<NPerThread, NumWarps, NThreadPerWarp>,
                                             sequence<KThreadPerBlock, KPerThread>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackV()
    {
        // TODO: this is for 3d layout
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
        return 16 / sizeof(VDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVnewDramTileDistribution()
    {
        using VLayout   = remove_cvref_t<typename Problem::VLayout>;
        using VDataType = remove_cvref_t<typename Problem::VDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::kN1;
        constexpr index_t kKPerBlock = Problem::kN0;

        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {

            constexpr index_t NPerThread      = 16 / sizeof(VDataType);
            constexpr index_t NThreadPerBlock = kNPerBlock / NPerThread;
            constexpr index_t KThreadPerWarp  = get_warp_size() / NThreadPerBlock;
            constexpr index_t NumWarps        = kBlockSize / get_warp_size();
            constexpr index_t KPerThread      = kKPerBlock / (NumWarps * KThreadPerWarp);

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<NThreadPerBlock, NPerThread>,
                                                 sequence<KPerThread, NumWarps, KThreadPerWarp>>,
                                           tuple<sequence<2>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<0, 2>>,
                                           sequence<1, 2>,
                                           sequence<1, 0>>{});
        }
        else
        {
            constexpr index_t KPerThread      = 16 / sizeof(VDataType);
            constexpr index_t KThreadPerBlock = kKPerBlock / KPerThread;
            constexpr index_t NThreadPerWarp  = get_warp_size() / KThreadPerBlock;
            constexpr index_t NumWarps        = kBlockSize / get_warp_size();
            constexpr index_t NPerThread      = kNPerBlock / (NumWarps * NThreadPerWarp);

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<NPerThread, NumWarps, NThreadPerWarp>,
                                                 sequence<KThreadPerBlock, KPerThread>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
        }
    }

    template <typename Problem, bool IsRotaryCosSinForQ>
    CK_TILE_HOST_DEVICE static constexpr auto GetRotaryCosSinTileSize()
    {
        constexpr index_t height = (IsRotaryCosSinForQ ? Problem::kM0 : Problem::kN0);

        if constexpr(Problem::RotaryEnum == RotaryEmbeddingEnum::HALF_ROTATED)
        {
            return make_tuple(number<height>{}, number<Problem::kK0>{});
        }
        else
        {
            return make_tuple(number<height>{}, number<Problem::kK0 / 2>{});
        }
    }

    template <typename Problem, bool IsRotaryCosSinForQ>
    CK_TILE_HOST_DEVICE static constexpr auto MakeRotaryCosSinTileDistribution()
    {
        using DataType = std::conditional_t<IsRotaryCosSinForQ,
                                            typename Problem::QDataType,
                                            typename Problem::KDataType>;

        constexpr auto TileSize = GetRotaryCosSinTileSize<Problem, IsRotaryCosSinForQ>();

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = TileSize[number<0>{}];
        constexpr index_t kKPerBlock = TileSize[number<1>{}];

        constexpr index_t KPerThread = []() {
            if constexpr(Problem::RotaryEnum == RotaryEmbeddingEnum::HALF_ROTATED)
            {
                /// NOTICE: we might need to lower down this to support smaller rotary_dim
                return 16 / sizeof(DataType);
            }
            else
            {
                return 8 / sizeof(DataType);
            }
        }();
        constexpr index_t KThreadPerBlock = kKPerBlock / KPerThread;
        constexpr index_t NThreadPerWarp  = get_warp_size() / KThreadPerBlock;
        constexpr index_t NumWarps        = kBlockSize / get_warp_size();
        constexpr index_t NPerThread      = kNPerBlock / (NumWarps * NThreadPerWarp);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<NPerThread, NumWarps, NThreadPerWarp>,
                                             sequence<KThreadPerBlock, KPerThread>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }
};

} // namespace ck_tile
