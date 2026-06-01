// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"

namespace ck_tile {

struct BlockFmhaPipelineQRKSVSWholeKPrefetchDefaultPolicy
    : BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                          /* AsyncCopy = */ false,
                                          /* NumPrefetchK = */ -1,
                                          /* NumPrefetchV = */ 2>
{
    static constexpr index_t NumPrefetchV = 2;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t IsPreloadWholeNextIterationK()
    {
        return Problem::BlockFmhaShape::kM0 <= 64;
    };

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetNumKLdsBuffers()
    {
        return 2;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto GetNumPrefetchV()
    {
        using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;

        constexpr index_t kN0 = BlockFmhaShape::kN0;
        constexpr index_t kK1 = BlockFmhaShape::kK1;

        constexpr index_t k1_loops = kN0 / kK1;

        return min(NumPrefetchV, k1_loops);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetNumVLdsBuffers()
    {
        return 2;
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;

        return BlockGemm::template MakeABlockTileDistribution<
            Problem::BlockFmhaShape::kM0,
            Problem::BlockFmhaShape::kQKHeaddim>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackK()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
        return 8 / sizeof(KDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr index_t NumKLdsBuffers = GetNumKLdsBuffers<Problem>();
        constexpr index_t kNPerBlock     = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock     = Problem::BlockFmhaShape::kK0;
        constexpr index_t kKPack         = GetSmemKPackK<Problem>();
        constexpr index_t kKVector       = GetAlignmentK<Problem>();

        static_assert(kKVector % kKPack == 0);

        constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<NumKLdsBuffers>{},
                       number<kKPerBlock / kKVector>{},
                       number<kKVector / kKPack>{},
                       number<kNPerBlock>{},
                       number<kKPack>{}),
            make_tuple(number<kKPerBlock * kNPerBlock + kKPerBlock * kKPack / kKVector>{},
                       number<kNPerBlock * kKVector + kKPack>{},
                       number<kNPerBlock * kKPack>{},
                       number<kKPack>{},
                       number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto k_lds_block_desc = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(number<NumKLdsBuffers>{}, number<kNPerBlock>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / kKVector>{},
                                                number<kKVector / kKPack>{},
                                                number<kKPack>{}))),
            make_tuple(sequence<0, 3>{}, sequence<1, 2, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return k_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;

        constexpr index_t MaxVectorSize = 16 / sizeof(KDataType);

        constexpr index_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        constexpr index_t kMaxVecLoad = min(ElemPerThread, MaxVectorSize);

        constexpr index_t KPerThread     = kMaxVecLoad;
        constexpr index_t KThreads       = kKPerBlock / KPerThread;
        constexpr index_t NThreadPerWarp = get_warp_size() / KThreads;
        constexpr index_t NumWarps       = kBlockSize / get_warp_size();
        constexpr index_t NPerThread     = kNPerBlock / (NThreadPerWarp * NumWarps);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<NPerThread, NThreadPerWarp, NumWarps>,
                                             sequence<KThreads, KPerThread>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<2>, sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        using VDataType = remove_cvref_t<typename Problem::VDataType>;

        constexpr index_t NumVLdsBuffers = GetNumVLdsBuffers<Problem>();

        constexpr index_t Banks        = get_n_lds_banks();
        constexpr index_t PixelsPerRow = Banks * 4 / sizeof(VDataType);
        constexpr index_t kKPack       = GetSmemKPackV<Problem>();
        static_assert(PixelsPerRow % kKPack == 0);
        constexpr index_t NPerRow    = PixelsPerRow / kKPack;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        static_assert(kNPerBlock % NPerRow == 0);
        static_assert(kKPerBlock % kKPack == 0);

        constexpr index_t VSingleSmemElementSpaceSize =
            (kKPerBlock / kKPack) * (kNPerBlock / NPerRow) * (PixelsPerRow + kKPack);

        constexpr auto v_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<NumVLdsBuffers>{},
                       number<kKPerBlock / kKPack>{},
                       number<kNPerBlock / NPerRow>{},
                       number<NPerRow>{},
                       number<kKPack>{}),
            make_tuple(number<VSingleSmemElementSpaceSize>{},
                       number<(kNPerBlock / NPerRow) * (PixelsPerRow + kKPack)>{},
                       number<PixelsPerRow + kKPack>{},
                       number<kKPack>{},
                       number<1>{}),
            number<kKPack>{},
            number<1>{});

        constexpr auto v_lds_block_desc = transform_tensor_descriptor(
            v_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(
                    number<NumVLdsBuffers>{}, number<kNPerBlock / NPerRow>{}, number<NPerRow>{})),
                make_merge_transform(make_tuple(number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
            make_tuple(sequence<0, 2, 3>{}, sequence<1, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return v_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        using VLayout = remove_cvref_t<typename Problem::BlockFmhaShape::VLayout>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t N1 = GetAlignmentV<Problem>();
            constexpr index_t N0 = kNPerBlock / N1; // P

            constexpr index_t ElemPerThread = kNPerBlock * kKPerBlock / kBlockSize;
            static_assert(ElemPerThread % N1 == 0);
            constexpr index_t K3     = ElemPerThread / N1;
            constexpr index_t kKPack = GetSmemKPackV<Problem>();
            static_assert(kKPack % K3 == 0);
            constexpr index_t K2 = kKPack / K3;
            if constexpr(get_warp_size() % (K2 * N0) == 0)
            {
                constexpr index_t K1 = get_warp_size() / (K2 * N0);
                constexpr index_t K0 = kBlockSize / get_warp_size();
                static_assert(kKPerBlock == K0 * K1 * K2 * K3);
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<N0, N1>, sequence<K0, K1, K2, K3>>,
                                               tuple<sequence<2>, sequence<2, 1, 2>>,
                                               tuple<sequence<0>, sequence<1, 0, 2>>,
                                               sequence<2, 1>,
                                               sequence<3, 1>>{});
            }
            else
            {
                constexpr index_t K1   = (K2 * N0) / get_warp_size();
                constexpr index_t K2_m = K2 / K1;
                constexpr index_t K0   = kBlockSize / get_warp_size() / K1;
                static_assert(kKPerBlock == K0 * K1 * K2_m * K3);
                return make_static_tile_distribution(
                    tile_distribution_encoding<sequence<1>,
                                               tuple<sequence<N0, N1>, sequence<K0, K1, K2_m, K3>>,
                                               tuple<sequence<2, 2>, sequence<1, 2>>,
                                               tuple<sequence<0, 1>, sequence<0, 2>>,
                                               sequence<2, 1>,
                                               sequence<3, 1>>{});
            }
        }
        else
        {
            constexpr index_t K1 = GetAlignmentV<Problem>();
            constexpr index_t K0 = kKPerBlock / K1;
            constexpr index_t N2 = get_warp_size() / K0;
            constexpr index_t N1 = kBlockSize / get_warp_size();
            static_assert(N2 != 0, "N2 is zero, which will lead to a division by zero error.");
            static_assert(N1 != 0, "N1 is zero, which will lead to a division by zero error.");
            constexpr index_t N0 = kNPerBlock / (N2 * N1);
            static_assert(N0 != 0);

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
        }
    }

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
            return BlockGemmARegBSmemCRegV2<GemmProblem, BlockGemmPolicy>{};
        else
            return BlockGemmARegBSmemCRegOneWarpV1<GemmProblem, BlockGemmPolicy>{};
    }

    // leave some exclusive space so that the second v_lds buffer will nenver overlap with the first
    // k_lds bufffer
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetExclusiveKLdsBytes()
    {
        constexpr index_t single_k_lds_buffer_size =
            GetSmemSizeK<Problem>() / GetNumKLdsBuffers<Problem>();
        constexpr index_t single_v_lds_buffer_size =
            GetSmemSizeV<Problem>() / GetNumVLdsBuffers<Problem>();

        if constexpr(single_k_lds_buffer_size <= single_v_lds_buffer_size)
            return 0;
        else
            return integer_least_multiple(single_k_lds_buffer_size - single_v_lds_buffer_size, 64);
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t IsFirstKLdsBufferOverlapLastVLdsBuffer()
    {
        using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;

        constexpr index_t k1_loops          = BlockFmhaShape::kN0 / BlockFmhaShape::kK1;
        constexpr index_t num_k_lds_buffers = GetNumKLdsBuffers<Problem>();
        constexpr index_t num_v_lds_buffers = GetNumVLdsBuffers<Problem>();

        constexpr index_t last_v_lds_buffer_offset =
            MakeVLdsBlockDescriptor<Problem>().get_element_space_size() / num_v_lds_buffers *
            ((k1_loops - 1) % num_v_lds_buffers) * sizeof(typename Problem::VDataType);

        constexpr index_t first_k_lds_buffer_size =
            MakeKLdsBlockDescriptor<Problem>().get_element_space_size() / num_k_lds_buffers *
            sizeof(typename Problem::KDataType);

        return GetExclusiveKLdsBytes<Problem>() + last_v_lds_buffer_offset <
               first_k_lds_buffer_size;
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeK()
    {
        return MakeKLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::KDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeV()
    {
        return MakeVLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::VDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        // assume V can reuse the other shared memory by K except the first
        // assume Dropout can reuse the shared memory by V
        return GetExclusiveKLdsBytes<Problem>() +
               max(GetSmemSizeK<Problem>() - GetExclusiveKLdsBytes<Problem>(),
                   max(GetSmemSizeV<Problem>(), GetSmemSizeDropout<Problem>(0)));
    }
};

} // namespace ck_tile
