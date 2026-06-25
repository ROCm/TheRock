// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2_custom_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v2.hpp"

// can remove all bank conflicts, but drop the performance for some cases
// Probably it is limited by compiler optimization.
#define CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD 0
namespace ck_tile {
// This pipeline is qkv all located in LDS
struct BlockFmhaPipelineQRKSVSAsyncTrloadDefaultPolicy
    : BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                          /* AsyncCopy = */ false,
                                          /* NumPrefetchK = */ 1,
                                          /* NumPrefetchV = */ 1>
{
    using BasePolicy = BlockFmhaPipelineQXKSVSCustomPolicy</* QLoadOnce = */ true,
                                                           /* AsyncCopy = */ false,
                                                           /* NumPrefetchK = */ 1,
                                                           /* NumPrefetchV = */ 1>;

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::QDataType);

        // this should align with MakeQDramTileDistribution()
        constexpr index_t ElemPerThread = (kMPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        return min(ElemPerThread, MaxVectorSize);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentOacc()
    {
        using OaccDataType = remove_cvref_t<typename Problem::OaccDataType>;

        return static_cast<index_t>(16 / sizeof(OaccDataType));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentK()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::KDataType);

        constexpr index_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        return min(ElemPerThread, MaxVectorSize);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentV()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::VDataType);

        constexpr index_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        return min(ElemPerThread, MaxVectorSize);
    }

    template <typename Problem, bool BypassLDS = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQDramTileDistribution()
    {
        if constexpr(!BypassLDS)
        {
            constexpr index_t kBlockSize = Problem::kBlockSize;
            constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

            constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::QDataType);

            constexpr index_t ElemPerThread = (kMPerBlock * kKPerBlock) / kBlockSize;
            static_assert(0 < ElemPerThread);
            constexpr index_t kMaxVecLoad = min(ElemPerThread, MaxVectorSize);

            constexpr index_t KPerThread     = kMaxVecLoad;
            constexpr index_t KThreads       = kKPerBlock / KPerThread;
            constexpr index_t MThreadPerWarp = get_warp_size() / KThreads;
            constexpr index_t NumWarps       = kBlockSize / get_warp_size();
            constexpr index_t MPerThread     = kMPerBlock / (MThreadPerWarp * NumWarps);

            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<MPerThread, NumWarps, MThreadPerWarp>,
                                                 sequence<KThreads, KPerThread>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
        }
        else
        {
            using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
            constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
            using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

            constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<0>{});
            constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{});

            constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
            constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

            constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
            constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

            constexpr auto q_block_outer_dstr_encoding = tile_distribution_encoding<
                sequence<NWarp>,
                tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                tuple<sequence<1, 0>>,
                tuple<sequence<1, 0>>,
                sequence<2, 1>,
                sequence<0, 0>>{};

            constexpr auto q_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
                q_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

            constexpr auto q_block_dstr = make_static_tile_distribution(q_block_dstr_encode);

            return q_block_dstr;
        }
    }

    template <typename Problem, bool LoadOnce = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock =
            LoadOnce ? Problem::BlockFmhaShape::kSubQKHeaddim : Problem::BlockFmhaShape::kK0;

        constexpr index_t MaxVectorSize = 16 / sizeof(KDataType);
        constexpr index_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;

        constexpr index_t K1 = min(MaxVectorSize, ElemPerThread);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;
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

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQRegTileDistribution()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{});

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

        constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        // Read M first, then K
        // This is the same data consume order as BlockGEMM
        constexpr auto q_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto q_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            q_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

        constexpr auto q_block_dstr = make_static_tile_distribution(q_block_dstr_encode);

        return q_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPackQ()
    {
        // TODO: this is for 3d layout
        using QDataType = remove_cvref_t<typename Problem::QDataType>;
        return static_cast<index_t>(16 / sizeof(QDataType));
    }

    template <typename Problem, bool Xor = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kSubQKHeaddim;

        constexpr index_t kKPack = GetSmemKPackQ<Problem>();

        constexpr auto q_lds_block_desc = [&]() {
            if constexpr(Xor)
            {
#if CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD
                constexpr auto LDSLayerSize  = 256 / sizeof(typename Problem::QDataType);
                constexpr auto XorLengthFold = LDSLayerSize / kKPerBlock;

                if constexpr(XorLengthFold > 1)
                {
                    constexpr auto q_lds_block_desc_naive = make_naive_tensor_descriptor(
                        make_tuple(number<kMPerBlock / XorLengthFold>{},
                                   number<LDSLayerSize / kKPack>{},
                                   number<kKPack>{}),
                        make_tuple(number<LDSLayerSize>{}, number<kKPack>{}, number<1>{}),
                        number<kKPack>{},
                        number<1>{});

                    constexpr auto q_lds_block_desc_permuted = transform_tensor_descriptor(
                        q_lds_block_desc_naive,
                        make_tuple(
                            make_xor_transform(make_tuple(number<kMPerBlock / XorLengthFold>{},
                                                          number<LDSLayerSize / kKPack>{})),
                            make_pass_through_transform(number<kKPack>{})),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}));

                    constexpr auto q_lds_block_desc_tmp = transform_tensor_descriptor(
                        q_lds_block_desc_permuted,
                        make_tuple(
                            make_pass_through_transform(number<kMPerBlock / XorLengthFold>{}),
                            make_unmerge_transform(
                                make_tuple(number<XorLengthFold>{}, number<kKPerBlock / kKPack>{})),
                            make_pass_through_transform(number<kKPack>{})),
                        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                        make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

                    return transform_tensor_descriptor(
                        q_lds_block_desc_tmp,
                        make_tuple(
                            make_merge_transform_v3_division_mod(make_tuple(
                                number<kMPerBlock / XorLengthFold>{}, number<XorLengthFold>{})),
                            make_merge_transform_v3_division_mod(
                                make_tuple(number<kMPerBlock / kKPack>{}, number<kKPack>{}))),
                        make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                        make_tuple(sequence<0>{}, sequence<1>{}));
                }
                else
#endif // CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD
                {
                    constexpr auto q_lds_block_desc_naive = make_naive_tensor_descriptor(
                        make_tuple(
                            number<kMPerBlock>{}, number<kKPerBlock / kKPack>{}, number<kKPack>{}),
                        make_tuple(number<kKPerBlock>{}, number<kKPack>{}, number<1>{}),
                        number<kKPack>{},
                        number<1>{});

                    constexpr auto q_lds_block_desc_permuted = transform_tensor_descriptor(
                        q_lds_block_desc_naive,
                        make_tuple(make_xor_transform(make_tuple(number<kMPerBlock>{},
                                                                 number<kKPerBlock / kKPack>{})),
                                   make_pass_through_transform(number<kKPack>{})),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}));

                    return transform_tensor_descriptor(
                        q_lds_block_desc_permuted,
                        make_tuple(make_pass_through_transform(number<kMPerBlock>{}),
                                   make_merge_transform_v3_division_mod(make_tuple(
                                       number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
                        make_tuple(sequence<0>{}, sequence<1, 2>{}),
                        make_tuple(sequence<0>{}, sequence<1>{}));
                }
            }
            else
            {
                return make_naive_tensor_descriptor(
                    make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
                    make_tuple(number<kKPerBlock>{}, number<1>{}),
                    number<kKPack>{},
                    number<1>{});
            }
        }();

        return q_lds_block_desc;
    }

    template <typename Problem, bool LoadOnce = false, bool Xor = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock =
            LoadOnce ? Problem::BlockFmhaShape::kSubQKHeaddim : Problem::BlockFmhaShape::kK0;

        constexpr index_t kKPack = GetSmemKPackK<Problem>();

        constexpr auto k_lds_block_desc = [&]() {
            if constexpr(Xor)
            {
#if CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD
                constexpr auto LDSLayerSize  = 256 / sizeof(typename Problem::KDataType);
                constexpr auto XorLengthFold = LDSLayerSize / kKPerBlock;

                if constexpr(XorLengthFold > 1)
                {
                    constexpr auto k_lds_block_desc_naive = make_naive_tensor_descriptor(
                        make_tuple(number<kNPerBlock / XorLengthFold>{},
                                   number<LDSLayerSize / kKPack>{},
                                   number<kKPack>{}),
                        make_tuple(number<LDSLayerSize>{}, number<kKPack>{}, number<1>{}),
                        number<kKPack>{},
                        number<1>{});

                    constexpr auto k_lds_block_desc_permuted = transform_tensor_descriptor(
                        k_lds_block_desc_naive,
                        make_tuple(
                            make_xor_transform(make_tuple(number<kNPerBlock / XorLengthFold>{},
                                                          number<LDSLayerSize / kKPack>{})),
                            make_pass_through_transform(number<kKPack>{})),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}));

                    constexpr auto k_lds_block_desc_tmp = transform_tensor_descriptor(
                        k_lds_block_desc_permuted,
                        make_tuple(
                            make_pass_through_transform(number<kNPerBlock / XorLengthFold>{}),
                            make_unmerge_transform(
                                make_tuple(number<XorLengthFold>{}, number<kKPerBlock / kKPack>{})),
                            make_pass_through_transform(number<kKPack>{})),
                        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                        make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

                    return transform_tensor_descriptor(
                        k_lds_block_desc_tmp,
                        make_tuple(
                            make_merge_transform_v3_division_mod(make_tuple(
                                number<kNPerBlock / XorLengthFold>{}, number<XorLengthFold>{})),
                            make_merge_transform_v3_division_mod(
                                make_tuple(number<kNPerBlock / kKPack>{}, number<kKPack>{}))),
                        make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                        make_tuple(sequence<0>{}, sequence<1>{}));
                }
                else
#endif // CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD
                {
                    constexpr auto k_lds_block_desc_naive = make_naive_tensor_descriptor(
                        make_tuple(
                            number<kNPerBlock>{}, number<kKPerBlock / kKPack>{}, number<kKPack>{}),
                        make_tuple(number<kKPerBlock>{}, number<kKPack>{}, number<1>{}),
                        number<kKPack>{},
                        number<1>{});

                    constexpr auto k_lds_block_desc_permuted = transform_tensor_descriptor(
                        k_lds_block_desc_naive,
                        make_tuple(make_xor_transform(make_tuple(number<kNPerBlock>{},
                                                                 number<kKPerBlock / kKPack>{})),
                                   make_pass_through_transform(number<kKPack>{})),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}));

                    return transform_tensor_descriptor(
                        k_lds_block_desc_permuted,
                        make_tuple(make_pass_through_transform(number<kNPerBlock>{}),
                                   make_merge_transform_v3_division_mod(make_tuple(
                                       number<kKPerBlock / kKPack>{}, number<kKPack>{}))),
                        make_tuple(sequence<0>{}, sequence<1, 2>{}),
                        make_tuple(sequence<0>{}, sequence<1>{}));
                }
            }
            else
            {
                return make_naive_tensor_descriptor(
                    make_tuple(number<kNPerBlock>{}, number<kKPerBlock>{}),
                    make_tuple(number<kKPerBlock>{}, number<1>{}),
                    number<kKPack>{},
                    number<1>{});
            }
        }();

        return k_lds_block_desc;
    }

    template <typename Problem, bool Xor = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t kKPack = GetSmemKPackV<Problem>();

        constexpr auto v_lds_block_desc = [&]() {
            if constexpr(Xor)
            {
                constexpr auto XorGroupSize =
                    Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{});

#if CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD
                constexpr auto LDSLayerSize  = 256 / sizeof(typename Problem::VDataType);
                constexpr auto XorLengthFold = LDSLayerSize / kNPerBlock;

                if constexpr(XorLengthFold > 1)
                {
                    constexpr auto v_lds_block_desc_naive = make_naive_tensor_descriptor(
                        make_tuple(number<kKPerBlock / XorLengthFold>{},
                                   number<LDSLayerSize / XorGroupSize>{},
                                   number<XorGroupSize>{}),
                        make_tuple(number<LDSLayerSize>{}, number<XorGroupSize>{}, number<1>{}),
                        number<kKPack>{},
                        number<1>{});

                    constexpr auto v_lds_block_desc_permuted = transform_tensor_descriptor(
                        v_lds_block_desc_naive,
                        make_tuple(
                            make_xor_transform(make_tuple(number<kKPerBlock / XorLengthFold>{},
                                                          number<LDSLayerSize / XorGroupSize>{})),
                            make_pass_through_transform(number<XorGroupSize>{})),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}));

                    constexpr auto v_lds_block_desc_tmp = transform_tensor_descriptor(
                        v_lds_block_desc_permuted,
                        make_tuple(
                            make_pass_through_transform(number<kKPerBlock / XorLengthFold>{}),
                            make_unmerge_transform(make_tuple(number<XorLengthFold>{},
                                                              number<kNPerBlock / XorGroupSize>{})),
                            make_pass_through_transform(number<XorGroupSize>{})),
                        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                        make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

                    return transform_tensor_descriptor(
                        v_lds_block_desc_tmp,
                        make_tuple(
                            make_merge_transform_v3_division_mod(make_tuple(
                                number<kKPerBlock / XorLengthFold>{}, number<XorLengthFold>{})),
                            make_merge_transform_v3_division_mod(make_tuple(
                                number<kNPerBlock / XorGroupSize>{}, number<XorGroupSize>{}))),
                        make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                        make_tuple(sequence<0>{}, sequence<1>{}));
                }
                else
#endif // CK_TILE_FMHA_HANDLE_XOR_LENGTH_FOLD
                {
                    constexpr auto v_lds_block_desc_naive = make_naive_tensor_descriptor(
                        make_tuple(number<kKPerBlock>{},
                                   number<kNPerBlock / XorGroupSize>{},
                                   number<XorGroupSize>{}),
                        make_tuple(number<kNPerBlock>{}, number<XorGroupSize>{}, number<1>{}),
                        number<kKPack>{},
                        number<1>{});

                    constexpr auto v_lds_block_desc_permuted = transform_tensor_descriptor(
                        v_lds_block_desc_naive,
                        make_tuple(make_xor_transform(make_tuple(
                                       number<kKPerBlock>{}, number<kNPerBlock / XorGroupSize>{})),
                                   make_pass_through_transform(number<XorGroupSize>{})),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}),
                        make_tuple(sequence<0, 1>{}, sequence<2>{}));

                    return transform_tensor_descriptor(
                        v_lds_block_desc_permuted,
                        make_tuple(
                            make_pass_through_transform(number<kKPerBlock>{}),
                            make_merge_transform_v3_division_mod(make_tuple(
                                number<kNPerBlock / XorGroupSize>{}, number<XorGroupSize>{}))),
                        make_tuple(sequence<0>{}, sequence<1, 2>{}),
                        make_tuple(sequence<0>{}, sequence<1>{}));
                }
            }
            else
            {
                return make_naive_tensor_descriptor(
                    make_tuple(number<kKPerBlock>{}, number<kNPerBlock>{}),
                    make_tuple(number<kNPerBlock>{}, number<1>{}),
                    number<kKPack>{},
                    number<1>{});
            }
        }();

        return v_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::QDataType,
                             typename Problem::KDataType,
                             typename Problem::SaccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kN0,
                                                    Problem::BlockFmhaShape::kK0>,
                                           typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm0WarpTile>>;

        using WarpGemm = WarpGemmDispatcher<typename Problem::QDataType,
                                            typename Problem::KDataType,
                                            typename Problem::SaccDataType,
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<0>{}),
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<1>{}),
                                            Problem::BlockFmhaShape::Gemm0WarpTile::at(number<2>{}),
                                            true>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV2CustomPolicy<typename Problem::QDataType,
                                                typename Problem::KDataType,
                                                typename Problem::SaccDataType,
                                                typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                WarpGemm,
                                                GemmLoopOrder::MNK>;

        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetPVBlockGemm()
    {
        using GemmProblem =
            BlockGemmProblem<typename Problem::PDataType,
                             typename Problem::VDataType,
                             typename Problem::OaccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Problem::BlockFmhaShape::kM0,
                                                    Problem::BlockFmhaShape::kN1,
                                                    Problem::BlockFmhaShape::kK1>,
                                           typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                           typename Problem::BlockFmhaShape::Gemm1WarpTile>>;

        using WarpGemm =
            WarpGemmDispatcher<typename Problem::PDataType,
                               typename Problem::VDataType,
                               typename Problem::OaccDataType,
                               Problem::BlockFmhaShape::Gemm1WarpTile::at(number<0>{}),
                               Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}),
                               Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}),
                               true,
                               false,
                               false,
                               ((Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}) == 16 &&
                                 Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}) == 32) ||
                                (Problem::BlockFmhaShape::Gemm1WarpTile::at(number<1>{}) == 32 &&
                                 Problem::BlockFmhaShape::Gemm1WarpTile::at(number<2>{}) == 16))
                                   ? WGAttrNumAccessEnum::Double
                                   : WGAttrNumAccessEnum::Single>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV2CustomPolicy<typename Problem::PDataType,
                                                typename Problem::VDataType,
                                                typename Problem::OaccDataType,
                                                typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                                WarpGemm,
                                                GemmLoopOrder::KMN>;

        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKRegTileDistribution()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        // Read N first, then K
        // This is the same data consume order as BlockGEMM
        constexpr auto k_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto k_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            k_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        constexpr auto k_block_dstr = make_static_tile_distribution(k_block_dstr_encode);

        return k_block_dstr;
    }

    template <typename Problem>
    CK_TILE_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t MaxVectorSize = 16 / sizeof(typename Problem::VDataType);

        constexpr index_t ElemPerThread = (kNPerBlock * kKPerBlock) / kBlockSize;
        static_assert(0 < ElemPerThread);
        constexpr index_t kMaxVecLoad = min(ElemPerThread, MaxVectorSize);

        constexpr index_t NPerThread     = kMaxVecLoad;
        constexpr index_t NThreads       = kNPerBlock / NPerThread;
        constexpr index_t KThreadPerWarp = get_warp_size() / NThreads;
        constexpr index_t NumWarps       = kBlockSize / get_warp_size();
        constexpr index_t KPerThread     = kKPerBlock / (KThreadPerWarp * NumWarps);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<KPerThread, NumWarps, KThreadPerWarp>,
                                             sequence<NThreads, NPerThread>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakePRegTileDistribution()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetPVBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<1>{});

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;

        constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WarpGemm::kM);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        // Read M first, then K
        // This is the same data consume order as BlockGEMM
        constexpr auto p_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<MIterPerWarp, MWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<1, 0>>,
                                       tuple<sequence<1, 0>>,
                                       sequence<2, 1>,
                                       sequence<0, 0>>{};

        constexpr auto p_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            p_block_outer_dstr_encoding, typename WarpGemm::AWarpDstrEncoding{});

        constexpr auto p_block_dstr = make_static_tile_distribution(p_block_dstr_encode);

        return p_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVRegTileDistribution()
    {
        using BlockGemm       = remove_cvref_t<decltype(GetPVBlockGemm<Problem>())>;
        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WarpGemm        = remove_cvref_t<decltype(config.template at<0>())>;

        constexpr index_t MWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<0>{});
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<1>{});

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WarpGemm::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WarpGemm::kK;

        // Read N first, then K
        // This is the same data consume order as BlockGEMM
        constexpr auto v_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<MWarp>,
                                       tuple<sequence<NIterPerWarp, NWarp>, sequence<KIterPerWarp>>,
                                       tuple<sequence<0, 1>>,
                                       tuple<sequence<0, 1>>,
                                       sequence<2, 1>,
                                       sequence<0, 0>>{};

        constexpr auto v_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            v_block_outer_dstr_encoding, typename WarpGemm::BWarpDstrEncoding{});

        constexpr auto v_block_dstr =
            make_static_tile_distribution(typename InputTileDistributionTraits<
                                          decltype(v_block_dstr_encode),
                                          typename Problem::VDataType>::TransposedDstrEncode{});

        return v_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemNPackS()
    {
        using SDataType = remove_cvref_t<typename Problem::SaccDataType>;
        return static_cast<index_t>(16 / sizeof(SDataType));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kNPack     = GetSmemNPackS<Problem>();

        constexpr auto s_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock / kNPack>{}, number<kMPerBlock>{}, number<kNPack>{}),
            make_tuple(number<(kMPerBlock + 1) * kNPack>{}, number<kNPack>{}, number<1>{}),
            number<kNPack>{},
            number<1>{});

        constexpr auto s_lds_block_desc = transform_tensor_descriptor(
            s_lds_block_desc_0,
            make_tuple(
                make_pass_through_transform(number<kMPerBlock>{}),
                make_merge_transform(make_tuple(number<kNPerBlock / kNPack>{}, number<kNPack>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return s_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSRegTileDistribution()
    {
        using BlockGemm = remove_cvref_t<decltype(GetKVBlockGemm<Problem>())>;

        constexpr auto config   = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG                = remove_cvref_t<decltype(config.template at<0>())>;
        constexpr index_t MWarp = config.template at<1>();
        constexpr index_t NWarp = config.template at<2>();

        // static_assert(MWarp == 1, "Check failed!");

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kTileK     = Problem::BlockFmhaShape::kN0;

        // K2 is equal to Impl::kABKPerLane * kKIterPerWarpGemm
        constexpr index_t K3 = WG::kK / WG::WarpGemmAttribute::Impl::kABKLane;
        constexpr index_t K2 = WG::WarpGemmAttribute::Impl::kABKLane;
        constexpr index_t K1 = kKPerBlock / (K2 * K3);
        constexpr index_t K0 = kTileK / kKPerBlock;
        constexpr index_t M2 = WG::WarpGemmAttribute::Impl::kAMLane;
        constexpr index_t M1 = MWarp;
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        constexpr auto s2_block_dstr_encoding =
            tile_distribution_encoding<sequence<NWarp>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1, K2, K3>>,
                                       tuple<sequence<1, 0>, sequence<2, 1>>,
                                       tuple<sequence<1, 0>, sequence<2, 2>>,
                                       sequence<1, 2, 2, 2>,
                                       sequence<0, 0, 1, 3>>{};

        constexpr auto s2_block_dstr = make_static_tile_distribution(s2_block_dstr_encoding);

        return s2_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeQ()
    {
        return MakeQLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::QDataType);
    }

    template <typename Problem, bool LoadOnce = false>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeK()
    {
        return MakeKLdsBlockDescriptor<Problem, LoadOnce>().get_element_space_size() *
               sizeof(typename Problem::KDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeV()
    {
        return MakeVLdsBlockDescriptor<Problem>().get_element_space_size() *
               sizeof(typename Problem::VDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeS()
    {
        constexpr index_t NWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{});

        return NWarp > 1 ? MakeSLdsBlockDescriptor<Problem>().get_element_space_size() *
                               sizeof(typename Problem::SaccDataType)
                         : 0;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        // Alignment on gfx950 is 1280 Bytes
        // Alignment before gfx950 is 512 Bytes.
        return max(GetSmemSizeQ<Problem>(),
                   GetSmemSizeK<Problem>() + GetSmemSizeS<Problem>() + GetSmemSizeV<Problem>());
    }
};

} // namespace ck_tile
