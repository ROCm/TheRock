// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fused_moe/pipeline/fused_moegemm_traits.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

namespace ck_tile {

struct FusedMoeGemmPipelineFlatmmPolicy
{
    CK_TILE_HOST_DEVICE static constexpr index_t GetAsyncCopyDwords()
    {
        // TODO: always 1 dword
        return 1;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignment_A()
    {
        // using async
        constexpr index_t copy_bytes = 4 * GetAsyncCopyDwords();
        constexpr index_t data_bytes = sizeof(typename Problem::ADataType);
        static_assert(copy_bytes % data_bytes == 0);
        return copy_bytes / data_bytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignment_G()
    {
        constexpr index_t copy_bytes = [&]() { return 16; }();
        constexpr index_t data_bytes = sizeof(typename Problem::GDataType);
        static_assert(copy_bytes % data_bytes == 0);
        return copy_bytes / data_bytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignment_D()
    {
        constexpr index_t copy_bytes = [&]() { return 16; }();
        constexpr index_t data_bytes = sizeof(typename Problem::DDataType);
        static_assert(copy_bytes % data_bytes == 0);
        return copy_bytes / data_bytes;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignment_O()
    {
        if constexpr(Problem::Traits::OAtomic == 1)
        {
            // pack fp16/bf16 atomic
            static_assert(sizeof(typename Problem::ODataType) == 2);
            return 2;
        }
        else if constexpr(Problem::Traits::OAtomic == 2)
        {
            // fp32 atomic
            return 1;
        }
        else
        {
            return 16 / sizeof(typename Problem::ODataType);
        }
    }

    template <typename DataType_>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPack()
    {
        // TODO: this is for 3d layout
        return 16 / sizeof(remove_cvref_t<DataType_>);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPack_A()
    {
        return GetSmemKPack<typename Problem::ADataType>();
    }

    // used for bridge LDS shuffle
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSmemKPack_Y()
    {
        // TODO: this should match mfma layout
        return 16 / sizeof(typename Problem::YDataType);
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize_A()
    {
        constexpr auto a_sld_desc = MakeLdsLoadDesc_A<Problem>();
        constexpr auto a_sst_desc = MakeLdsStoreDesc_A<Problem>();
        static_assert(a_sld_desc.get_element_space_size() == a_sst_desc.get_element_space_size());
        return a_sld_desc.get_element_space_size();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize_Bridge()
    {
        constexpr auto bridge_sld_desc = MakeBridgeLdsLoadDesc<Problem>();
        constexpr auto bridge_sst_desc = MakeBridgeLdsStoreDesc<Problem>();
        static_assert(bridge_sld_desc.get_element_space_size() ==
                      bridge_sst_desc.get_element_space_size());
        return bridge_sld_desc.get_element_space_size();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        constexpr index_t a_lds      = GetSmemSize_A<Problem>();
        constexpr index_t bridge_lds = GetSmemSize_Bridge<Problem>();
        return max(a_lds, bridge_lds);
    }

    template <index_t MPerBlock, index_t KPerBlock, index_t NumWarps, index_t Alignment>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_SimpleMxK()
    {
        constexpr index_t K_vec = Alignment;
        constexpr index_t K_rem = KPerBlock / K_vec;

        if constexpr(get_warp_size() < K_rem)
        {
            static_assert(K_rem % get_warp_size() == 0);
            constexpr index_t K_lan = get_warp_size(); // lane within same wave is along gemm-k
            constexpr index_t K_wav = K_rem / get_warp_size();
            static_assert(K_wav <= NumWarps, "not not support thread has repeat along K yet");
            constexpr index_t M_wav = NumWarps / K_wav;
            static_assert(MPerBlock % M_wav == 0, "this tile size is too small please check");
            constexpr index_t M_rep = MPerBlock / M_wav;

            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<1>,
                    tuple<sequence<M_rep, M_wav>, sequence<K_wav, K_lan, K_vec>>,
                    tuple<sequence<1, 2>, sequence<2>>,
                    tuple<sequence<1, 0>, sequence<1>>,
                    sequence<1, 2>,
                    sequence<0, 2>>{});
        }
        else
        {
            constexpr index_t K_lan = K_rem;
            constexpr index_t M_lan = get_warp_size() / K_lan;
            constexpr index_t M_wav = NumWarps;
            static_assert(MPerBlock % (M_lan * M_wav) == 0,
                          "this tile size is too small please check");
            constexpr index_t M_rep = MPerBlock / (M_lan * M_wav);
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<1>,
                    tuple<sequence<M_rep, M_wav, M_lan>, sequence<K_lan, K_vec>>,
                    tuple<sequence<1>, sequence<1, 2>>,
                    tuple<sequence<1>, sequence<2, 0>>,
                    sequence<1, 2>,
                    sequence<0, 1>>{});
        }
    }

    // optimized version for async, not same as simple MXK dist(pay attention!!)
    template <index_t MPerBlock, index_t KPerBlock, index_t NumWarps, index_t Alignment>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_SimpleMxK_Async()
    {
        constexpr index_t K_vec = Alignment;
        constexpr index_t K_rem = KPerBlock / K_vec;

        if constexpr(get_warp_size() <= K_rem)
        {
            static_assert(K_rem % get_warp_size() == 0);
            constexpr index_t K_lan = get_warp_size(); // lane within same wave is along gemm-k
            constexpr index_t K_wav = K_rem / get_warp_size();
            static_assert(K_wav <= NumWarps, "do not support thread has repeat along K yet");
            constexpr index_t M_wav = NumWarps / K_wav;
            static_assert(MPerBlock % M_wav == 0, "this tile size is too small please check");
            constexpr index_t M_rep = MPerBlock / M_wav;
            // NOTE: no swap, but hard to avoid LDS bank conflict
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<1>,
                    tuple<sequence<M_rep, M_wav>, sequence<K_wav, K_lan, K_vec>>,
                    tuple<sequence<1, 2>, sequence<2>>,
                    tuple<sequence<1, 0>, sequence<1>>,
                    sequence<1, 2>,
                    sequence<0, 2>>{});
        }
        else
        {
            constexpr index_t K_lan = K_rem;
            constexpr index_t M_lan = get_warp_size() / K_lan;
            constexpr index_t M_wav = NumWarps;
            static_assert(MPerBlock % (M_lan * M_wav) == 0,
                          "this tile size is too small please check");
            constexpr index_t M_rep = MPerBlock / (M_lan * M_wav);
            // NOTE: swapped for LDS load bank conflict free
            return make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<1>,
                    // Note M_wave(num waves) is the fastest dim, different from sipmle 2d
                    // distribution
                    tuple<sequence<M_rep, M_lan, M_wav>, sequence<K_lan, K_vec>>,
                    tuple<sequence<1>, sequence<1, 2>>,
                    tuple<sequence<2>, sequence<1, 0>>,
                    sequence<1, 2>,
                    sequence<0, 1>>{});
        }
    }

    template <index_t WarpPerBlock_N_,
              index_t WarpPerBlock_K_,
              index_t Repeat_N_,
              index_t Repeat_K_,
              index_t WarpSize_,
              index_t Alignment_>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_Nr_Kr_W()
    {
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<Repeat_N_, WarpPerBlock_N_>,
                                             sequence<Repeat_K_, WarpPerBlock_K_>,
                                             sequence<WarpSize_, Alignment_>>,
                                       tuple<sequence<1, 2>, sequence<3>>,
                                       tuple<sequence<1, 1>, sequence<0>>,
                                       sequence<1, 2, 3>,
                                       sequence<0, 0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_A()
    {
        constexpr index_t Block_M_   = Problem::BlockShape::Block_M0;
        constexpr index_t Block_K_   = Problem::BlockShape::Block_K0;
        constexpr index_t NumWarps_  = Problem::BlockShape::NumWarps;
        constexpr index_t Alignment_ = GetAlignment_A<Problem>();
        return MakeGlobalTileDistribution_SimpleMxK_Async<Block_M_,
                                                          Block_K_,
                                                          NumWarps_,
                                                          Alignment_>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_G()
    {
        constexpr auto PermuteEnum = Problem::Traits::PermuteEnum;
        // constexpr index_t hidden_radio_0 = Problem::Traits::IsGateOnly ? 1 : 2;
        using S_ = typename Problem::BlockShape;
        if constexpr(PermuteEnum == FusedMoeGemmWeightPermuteEnum::b_nr_kr_waveflatten)
        {
            // number<S_::WarpPerBlock_N0>{}.rrr();
            // number<S_::Repeat_N0>{}.eee();
            return MakeGlobalTileDistribution_Nr_Kr_W<S_::WarpPerBlock_N0,
                                                      S_::WarpPerBlock_K0,
                                                      S_::Repeat_N0, /// hidden_radio_0,
                                                      S_::Repeat_K0,
                                                      get_warp_size(),
                                                      GetAlignment_G<Problem>()>();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_D()
    {
        constexpr auto PermuteEnum = Problem::Traits::PermuteEnum;
        using S_                   = typename Problem::BlockShape;
        if constexpr(PermuteEnum == FusedMoeGemmWeightPermuteEnum::b_nr_kr_waveflatten)
        {
            return MakeGlobalTileDistribution_Nr_Kr_W<S_::WarpPerBlock_N1,
                                                      S_::WarpPerBlock_K1,
                                                      S_::Repeat_N1,
                                                      S_::Repeat_K1,
                                                      get_warp_size(),
                                                      GetAlignment_D<Problem>()>();
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeGlobalTileDistribution_O()
    {
        using S_       = remove_cvref_t<typename Problem::BlockShape>;
        using WarpGemm = remove_cvref_t<decltype(GetWarpGemm1<Problem>())>;
        // using CDataType = typename WarpGemm::CDataType;

        constexpr auto c_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<S_::Repeat_M1, S_::WarpPerBlock_M1>,
                                             sequence<S_::Repeat_N1, S_::WarpPerBlock_N1>>,
                                       tuple<sequence<1, 2>>,
                                       tuple<sequence<1, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        return c_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsStoreDesc_A()
    {
        // A async->LDS
        constexpr index_t Block_M = Problem::BlockShape::Block_M0;
        constexpr index_t Block_K = Problem::BlockShape::Block_K0;
        // constexpr index_t BlockSize = Problem::BlockShape::BlockSize;
        constexpr index_t WarpSize = ck_tile::get_warp_size();
        constexpr index_t NumWarps = Problem::BlockShape::NumWarps;

        constexpr index_t KPack   = GetSmemKPack_A<Problem>(); // LDS
        constexpr index_t KVector = GetAlignment_A<Problem>(); // async copy 1 dword
        constexpr index_t KPad    = KPack;                     // pad between warps

        static_assert(Block_K % KVector == 0);
        constexpr index_t LanesPerK = Block_K / KVector; // how many thread loading K
        if constexpr(LanesPerK >= WarpSize)
        {
            // need multiple waves to load K
            static_assert(LanesPerK % WarpSize == 0);
            constexpr index_t wavesPerK = LanesPerK / WarpSize;
            if constexpr(wavesPerK > NumWarps)
            {
                // TODO: need multiple issues along K to load all data
            }
            else
            {
                constexpr index_t wavesPerM     = NumWarps / wavesPerK;
                constexpr index_t NumIssues     = Block_M / wavesPerM;
                constexpr auto lds_block_desc_0 = make_naive_tensor_descriptor(
                    make_tuple(number<NumIssues>{},                             // m0
                               number<wavesPerM>{},                             // m1
                               number<wavesPerK>{},                             // k0
                               number<WarpSize>{},                              // k1
                               number<KVector>{}),                              // k2
                    make_tuple(number<NumWarps*(WarpSize * KVector + KPad)>{},  // m0
                               number<wavesPerK*(WarpSize * KVector + KPad)>{}, // m1
                               number<WarpSize * KVector + KPad>{},             // k0
                               number<KVector>{},                               // k1
                               number<1>{}),                                    // k2
                    number<KVector>{}, // lds store vector(actually no explicit store)
                    number<1>{});

                constexpr auto lds_block_desc_issues_warps_lanes = transform_tensor_descriptor(
                    lds_block_desc_0,
                    make_tuple(
                        make_pass_through_transform(number<NumIssues>{}),
                        make_merge_transform(make_tuple(number<wavesPerM>{}, number<wavesPerK>{})),
                        make_merge_transform(make_tuple(number<WarpSize>{}, number<KVector>{}))),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

                return lds_block_desc_issues_warps_lanes;
            }
        }
        else
        {
            // lanes within a wave load different M but same K
            static_assert(WarpSize % LanesPerK == 0);
            constexpr index_t LaneGroups = WarpSize / LanesPerK; // along m
            constexpr index_t NumIssues  = Block_M / (LaneGroups * NumWarps);

            constexpr auto lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<NumIssues>{},                            // m0
                           number<LaneGroups>{},                           // m1
                           number<NumWarps>{},                             // m2
                           number<LanesPerK>{},                            // k0
                           number<KVector>{}),                             // k1
                make_tuple(number<NumWarps*(WarpSize * KVector + KPad)>{}, // m0
                           number<Block_K>{},                              // m1
                           number<WarpSize * KVector + KPad>{},            // m2
                           number<KVector>{},                              // k0
                           number<1>{}),                                   // k1
                number<KVector>{}, // lds store vector(actually no explicit store)
                number<1>{});

            constexpr auto lds_block_desc_issues_warps_lanes = transform_tensor_descriptor(
                lds_block_desc_0,
                make_tuple(make_pass_through_transform(number<NumIssues>{}),
                           make_pass_through_transform(number<NumWarps>{}),
                           make_merge_transform(make_tuple(
                               number<LaneGroups>{}, number<LanesPerK>{}, number<KVector>{}))),
                make_tuple(sequence<0>{}, sequence<2>{}, sequence<1, 3, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

            return lds_block_desc_issues_warps_lanes;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsLoadDesc_A()
    {
        // A async->LDS
        // Note that, this descriptor is only to construct the layout inside LDS
        // in real Gemm pipeline, ds_read may not follow this pattern
        // (may follow that in tile_distribution)
        // below code is almost the same as SmemStore dist, with difference:
        //  1). modify the GuaranteedLastDimensionVectorLength of naive tensor desc
        //  2). return discriptor is in NxK 2d layout
        constexpr index_t Block_M = Problem::BlockShape::Block_M0;
        constexpr index_t Block_K = Problem::BlockShape::Block_K0;
        // constexpr index_t BlockSize = Problem::BlockShape::BlockSize;
        constexpr index_t WarpSize = ck_tile::get_warp_size();
        constexpr index_t NumWarps = Problem::BlockShape::NumWarps;

        constexpr index_t KPack   = GetSmemKPack_A<Problem>(); // LDS
        constexpr index_t KVector = GetAlignment_A<Problem>(); // async copy 1 dword
        constexpr index_t KPad    = KPack;                     // pad between warps

        static_assert(Block_K % KVector == 0);
        constexpr index_t LanesPerK = Block_K / KVector; // how many thread loading K
        if constexpr(LanesPerK >= WarpSize)
        {
            // need multiple waves to load K
            static_assert(LanesPerK % WarpSize == 0);
            constexpr index_t wavesPerK = LanesPerK / WarpSize;
            if constexpr(wavesPerK >= NumWarps)
            {
                // TODO: need multiple issues along K to load all data
            }
            else
            {
                constexpr index_t wavesPerM     = NumWarps / wavesPerK;
                constexpr index_t NumIssues     = Block_M / wavesPerM;
                constexpr auto lds_block_desc_0 = make_naive_tensor_descriptor(
                    make_tuple(number<NumIssues>{},                             // m0
                               number<wavesPerM>{},                             // m1
                               number<wavesPerK>{},                             // k0
                               number<WarpSize>{},                              // k1
                               number<KVector>{}),                              // k2
                    make_tuple(number<NumWarps*(WarpSize * KVector + KPad)>{},  // m0
                               number<wavesPerK*(WarpSize * KVector + KPad)>{}, // m1
                               number<WarpSize * KVector + KPad>{},             // k0
                               number<KVector>{},                               // k1
                               number<1>{}),                                    // k2
                    number<KPack>{},                                            // lds load vector
                    number<1>{});

                constexpr auto lds_desc_m_k = transform_tensor_descriptor(
                    lds_block_desc_0,
                    make_tuple(
                        make_merge_transform(make_tuple(number<NumIssues>{}, number<wavesPerM>{})),
                        make_merge_transform(make_tuple(
                            number<wavesPerK>{}, number<WarpSize>{}, number<KVector>{}))),
                    make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                return lds_desc_m_k;
            }
        }
        else
        {
            // lanes within a wave load different M but same K
            static_assert(WarpSize % LanesPerK == 0);
            constexpr index_t LaneGroups = WarpSize / LanesPerK; // along m
            constexpr index_t NumIssues  = Block_M / (LaneGroups * NumWarps);

            constexpr auto lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<NumIssues>{},                            // m0
                           number<LaneGroups>{},                           // m1
                           number<NumWarps>{},                             // m2
                           number<LanesPerK>{},                            // k0
                           number<KVector>{}),                             // k1
                make_tuple(number<NumWarps*(WarpSize * KVector + KPad)>{}, // m0
                           number<Block_K>{},                              // m1
                           number<WarpSize * KVector + KPad>{},            // m2
                           number<KVector>{},                              // k0
                           number<1>{}),                                   // k1
                number<KPack>{},                                           // lds load vector
                number<1>{});

            constexpr auto lds_desc_m_k = transform_tensor_descriptor(
                lds_block_desc_0,
                make_tuple(
                    make_merge_transform(
                        make_tuple(number<NumIssues>{}, number<LaneGroups>{}, number<NumWarps>{})),
                    make_merge_transform(make_tuple(number<LanesPerK>{}, number<KVector>{}))),
                make_tuple(sequence<0, 1, 2>{}, sequence<3, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return lds_desc_m_k;
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBridgeLdsLoadDesc()
    {
        constexpr index_t Block_M = Problem::BlockShape::Block_M0;
        constexpr index_t Block_N = Problem::BlockShape::Block_N0;

        constexpr index_t KVector = GetSmemKPack_Y<Problem>(); // async copy 1 dword
        constexpr index_t KPad    = 0;                         // pad between warps

        constexpr auto desc =
            make_naive_tensor_descriptor(make_tuple(number<Block_M>{}, number<Block_N>{}),
                                         make_tuple(number<Block_N + KPad>{}, number<1>{}),
                                         number<KVector>{},
                                         number<1>{});
        return desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBridgeLdsStoreDesc()
    {
        constexpr index_t Block_M = Problem::BlockShape::Block_M0;
        constexpr index_t Block_N = Problem::BlockShape::Block_N0;

        constexpr index_t KVector = GetSmemKPack_Y<Problem>(); // async copy 1 dword
        constexpr index_t KPad    = 0; // KVector;                   // pad between warps

        constexpr auto desc =
            make_naive_tensor_descriptor(make_tuple(number<Block_M>{}, number<Block_N>{}),
                                         make_tuple(number<Block_N + KPad>{}, number<1>{}),
                                         number<KVector>{},
                                         number<1>{});
        return desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBridgeLdsStoreForUKDesc()
    {
        constexpr index_t WarpPerBlock_N = Problem::BlockShape::WarpPerBlock_N0;
        constexpr index_t Repeat_N       = Problem::BlockShape::Repeat_N0;
        constexpr index_t Repeat_M       = Problem::BlockShape::Repeat_M0;

        constexpr index_t kAMLane     = 16;
        constexpr index_t kABKLane    = 4;
        constexpr index_t kABKPerLane = 4;

        constexpr index_t KPack = kABKPerLane;

        constexpr auto lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<Repeat_M>{},                                               // m
                       number<Repeat_N>{},                                               // n
                       number<WarpPerBlock_N>{},                                         // n
                       number<kABKLane>{},                                               // n
                       number<kAMLane>{},                                                // m
                       number<KPack>{}),                                                 // n
            make_tuple(number<Repeat_N * WarpPerBlock_N * kABKLane * kAMLane * KPack>{}, //  m
                       number<WarpPerBlock_N * kABKLane * kAMLane * KPack>{},            //  n
                       number<kABKLane * kAMLane * KPack>{},                             //  n
                       number<kAMLane * KPack>{},                                        //  n
                       number<KPack>{},                                                  //  m
                       number<1>{}),                                                     //  n
            number<KPack>{}, // lds store vector(actually no explicit store)
            number<1>{});

        constexpr auto desc = transform_tensor_descriptor(
            lds_block_desc_0,
            make_tuple(make_merge_transform(make_tuple(number<Repeat_M>{}, number<kAMLane>{})),
                       make_merge_transform(make_tuple(number<Repeat_N>{},
                                                       number<WarpPerBlock_N>{},
                                                       number<kABKLane>{},
                                                       number<KPack>{}))),
            make_tuple(sequence<0, 4>{}, sequence<1, 2, 3, 5>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemm0()
    {
        using S_ = typename Problem::BlockShape;
        // A is vgpr, B is agpr. But since we transposed, so also need swap this
        // TODO: this is ugly
        constexpr auto wg_ctrl = WGAttrCtlEnum::Raw_avv;
        // TODO: ugly
        if constexpr(std::is_same_v<typename Problem::ADataType, ck_tile::bf16_t> &&
                     std::is_same_v<typename Problem::GDataType, ck_tile::bf16_t> &&
                     S_::Warp_M0 == 32 && S_::Warp_N0 == 32 && S_::Warp_K0 == 16)
        {
            return WarpGemmImpl<WarpGemmAttributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
                WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<wg_ctrl>,
                2>>{};
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, ck_tile::int8_t> &&
                          std::is_same_v<typename Problem::GDataType, ck_tile::int8_t> &&
                          S_::Warp_M0 == 32 && S_::Warp_N0 == 32 && S_::Warp_K0 == 32)
        {
            return WarpGemmImpl<WarpGemmAttributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
                WarpGemmAttributeMfmaImpl_i32_32x32x16_i8<wg_ctrl>,
                2>>{};
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSequencer_0()
    {
        // this function return seq<...> used to identify gld/sld/valu... inside mfma sequence
        // the purpose is to hide thoes instructions under mfma
        // every value inside seq<...> is a mask, indicating a specific operation
        using S_                = typename Problem::BlockShape;
        constexpr index_t SLD_A = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::SLD_A);
        constexpr index_t GLD_A = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GLD_A);
        constexpr index_t GLD_B = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GLD_B);
        if constexpr(std::is_same_v<typename Problem::YDataType, ck_tile::bf16_t> &&
                     std::is_same_v<typename Problem::DDataType, ck_tile::bf16_t> &&
                     S_::Warp_M0 == 32 && S_::Warp_N0 == 32 && S_::Warp_K0 == 16 &&
                     S_::Block_M0 == 32 && S_::Block_N0 == 512 && S_::Block_K0 == 128 &&
                     S_::Block_N1 == 128)
        {
            // Total 64 instructions, 32 buffer-load-dwordx4 gld_b, 8x buffer-load-dwordx1-async
            // gld_a 8x ds_read_b128 sld_a total 64 slot :)
            // clang-format off
            constexpr auto seq_all =
                    //       0       1       2        3       4      5        6       7
                   sequence<GLD_B,  GLD_A,  GLD_B,  GLD_A,  GLD_B,  GLD_A,  GLD_B,  GLD_A,    // 0
                            GLD_B,  GLD_A,  GLD_B,  GLD_A,  GLD_B,  GLD_A,  GLD_B,  GLD_A,    // 1
                            GLD_B,  SLD_A,  GLD_B,  SLD_A,  GLD_B,  SLD_A,  GLD_B,  SLD_A,    // 2
                            GLD_B,  SLD_A,  GLD_B,  SLD_A,  GLD_B,  SLD_A,  GLD_B,  SLD_A,    // 3
                            GLD_B,      0,  GLD_B,      0,  GLD_B,      0,  GLD_B,      0,    // 4
                            GLD_B,      0,  GLD_B,      0,  GLD_B,      0,  GLD_B,      0,    // 5
                            GLD_B,      0,  GLD_B,      0,  GLD_B,      0,  GLD_B,      0,    // 6
                            GLD_B,      0,  GLD_B,      0,  GLD_B,      0,  GLD_B,      0>{}; // 7
            return seq_all;
            // clang-format on
        }
        else if constexpr(std::is_same_v<typename Problem::YDataType, ck_tile::bf16_t> &&
                          std::is_same_v<typename Problem::DDataType, ck_tile::bf16_t> &&
                          S_::Warp_M0 == 32 && S_::Warp_N0 == 32 && S_::Warp_K0 == 16 &&
                          S_::Block_M0 == 32 && S_::Block_N0 == 256 && S_::Block_K0 == 128 &&
                          S_::Block_N1 == 128)
        {
            // Total 32 instructions, 16 buffer-load-dwordx4 gld_b, 8x buffer-load-dwordx1-async
            // gld_a 8x ds_read_b128 sld_a total 64 slot :)
            // clang-format off
            constexpr auto seq_all =
                    //       0       1       2        3       4      5        6       7
                   sequence<GLD_B,  GLD_A,  GLD_B,  GLD_A,  GLD_B,  GLD_A,  GLD_B,  GLD_A,    // 0
                            GLD_B,  GLD_A,  GLD_B,  GLD_A,  GLD_B,  GLD_A,  GLD_B,  GLD_A,    // 1
                            GLD_B,  SLD_A,  GLD_B,  SLD_A,  GLD_B,  SLD_A,  GLD_B,  SLD_A,    // 2
                            GLD_B,  SLD_A,  GLD_B,  SLD_A,  GLD_B,  SLD_A,  GLD_B,  SLD_A>{};    // 3
            return seq_all;
            // clang-format on
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetSequencer_1()
    {
        // this function return seq<...> used to identify gld/sld/valu... inside mfma sequence
        // the purpose is to hide thoes instructions under mfma
        // every value inside seq<...> is a mask, indicating a specific operation
        using S_                = typename Problem::BlockShape;
        constexpr index_t GLD_B = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GLD_B);
        constexpr index_t GST_O = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GST_O);
        if constexpr(std::is_same_v<typename Problem::YDataType, ck_tile::bf16_t> &&
                     std::is_same_v<typename Problem::DDataType, ck_tile::bf16_t> &&
                     S_::Warp_M1 == 32 && S_::Warp_N1 == 32 && S_::Warp_K1 == 16 &&
                     S_::Block_M0 == 32 && S_::Block_N0 == 512 && S_::Block_K0 == 128 &&
                     S_::Block_N1 == 128)
        {
            // Total 64 instructions, 32 buffer-load-dwordx4 gld_b, 8x buffer-load-dwordx1-async
            // gld_a 8x ds_read_b128 sld_a total 64 slot :)
            // clang-format off
            constexpr auto seq_all =
                    //       0       1       2        3       4      5        6       7
                   sequence<GLD_B,  GST_O,  GLD_B,  GST_O,  GLD_B,  GST_O,  GLD_B,  GST_O,    // 0
                            GLD_B,  GST_O,  GLD_B,  GST_O,  GLD_B,  GST_O,  GLD_B,  GST_O,    // 1
                            GLD_B,      0,  GLD_B,      0,  GLD_B,      0,  GLD_B,      0,    // 2
                            GLD_B,      0,  GLD_B,      0,  GLD_B,      0,  GLD_B,      0,    // 3
                            GLD_B,      0,  GLD_B,      0,  GLD_B,      0,  GLD_B,      0,    // 4
                            GLD_B,      0,  GLD_B,      0,  GLD_B,      0,  GLD_B,      0,    // 5
                            GLD_B,      0,  GLD_B,      0,  GLD_B,      0,  GLD_B,      0,    // 6
                            GLD_B,      0,  GLD_B,      0,  GLD_B,      0,  GLD_B,      0>{}; // 7
            return seq_all;
            // clang-format on
        }
        else if constexpr(std::is_same_v<typename Problem::YDataType, ck_tile::bf16_t> &&
                          std::is_same_v<typename Problem::DDataType, ck_tile::bf16_t> &&
                          S_::Warp_M1 == 32 && S_::Warp_N1 == 32 && S_::Warp_K1 == 16 &&
                          S_::Block_M0 == 32 && S_::Block_N0 == 256 && S_::Block_K0 == 128 &&
                          S_::Block_N1 == 128)
        {
            // Total 64 instructions, 32 buffer-load-dwordx4 gld_b, 8x buffer-load-dwordx1-async
            // gld_a 8x ds_read_b128 sld_a total 64 slot :)
            // clang-format off
            constexpr auto seq_all =
                    //       0       1       2        3       4      5        6       7
                   sequence<GLD_B,  GST_O,  GLD_B,  GST_O,  GLD_B,  GST_O,  GLD_B,  GST_O,    // 0
                            GLD_B,  GST_O,  GLD_B,  GST_O,  GLD_B,  GST_O,  GLD_B,  GST_O,    // 1
                            GLD_B,      0,  GLD_B,      0,  GLD_B,      0,  GLD_B,      0,    // 2
                            GLD_B,      0,  GLD_B,      0,  GLD_B,      0,  GLD_B,      0>{};    // 3
            return seq_all;
            // clang-format on
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemm1()
    {
        using S_               = typename Problem::BlockShape;
        constexpr auto wg_ctrl = WGAttrCtlEnum::Raw_avv;
        // TODO: ugly
        if constexpr(std::is_same_v<typename Problem::YDataType, ck_tile::bf16_t> &&
                     std::is_same_v<typename Problem::DDataType, ck_tile::bf16_t> &&
                     S_::Warp_M0 == 32 && S_::Warp_N0 == 32 && S_::Warp_K0 == 16)
        {
            return WarpGemmImpl<WarpGemmAttributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
                WarpGemmAttributeMfmaImplBf16Bf16F32M32N32K8<wg_ctrl>,
                2>>{};
        }
        else if constexpr(std::is_same_v<typename Problem::YDataType, ck_tile::int8_t> &&
                          std::is_same_v<typename Problem::DDataType, ck_tile::int8_t> &&
                          S_::Warp_M0 == 32 && S_::Warp_N0 == 32 && S_::Warp_K0 == 32)
        {
            return WarpGemmImpl<WarpGemmAttributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
                WarpGemmAttributeMfmaImpl_i32_32x32x16_i8<wg_ctrl>,
                2>>{};
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeCBlockTile_Gemm0()
    {
        using S_        = remove_cvref_t<typename Problem::BlockShape>;
        using WarpGemm  = remove_cvref_t<decltype(GetWarpGemm0<Problem>())>;
        using CDataType = typename WarpGemm::CDataType;

        constexpr auto c_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<S_::Repeat_M0, S_::WarpPerBlock_M0>,
                                             sequence<S_::Repeat_N0, S_::WarpPerBlock_N0>>,
                                       tuple<sequence<1, 2>>,
                                       tuple<sequence<1, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);
        return c_block_tensor;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeCBlockTile_Gemm1()
    {
        using S_        = remove_cvref_t<typename Problem::BlockShape>;
        using WarpGemm  = remove_cvref_t<decltype(GetWarpGemm1<Problem>())>;
        using CDataType = typename WarpGemm::CDataType;

        constexpr auto c_block_outer_dstr_encoding =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<S_::Repeat_M1, S_::WarpPerBlock_M1>,
                                             sequence<S_::Repeat_N1, S_::WarpPerBlock_N1>>,
                                       tuple<sequence<1, 2>>,
                                       tuple<sequence<1, 1>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WarpGemm::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);
        return c_block_tensor;
    }

    // this is used as A matrix for 2nd gemm
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeYTileDistribution()
    {
        using S_       = remove_cvref_t<typename Problem::BlockShape>;
        using WarpGemm = remove_cvref_t<decltype(GetWarpGemm1<Problem>())>;

        // TODO: all waves a along different N, but same M
        constexpr auto y_outer_dstr_enc =
            tile_distribution_encoding<sequence<S_::WarpPerBlock_M1>,
                                       tuple<sequence<S_::Repeat_M1>, sequence<S_::Repeat_K1>>,
                                       tuple<sequence<0>>,
                                       tuple<sequence<0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{};

        constexpr auto y_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            y_outer_dstr_enc, typename WarpGemm::AWarpDstrEncoding{});
        constexpr auto y_block_dstr = make_static_tile_distribution(y_block_dstr_encode);
        return y_block_dstr;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeYBlockTile()
    {
        constexpr auto y_block_dstr = MakeYTileDistribution<Problem>();
        auto y_block_tensor =
            make_static_distributed_tensor<typename Problem::YDataType>(y_block_dstr);
        return y_block_tensor;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetUK_0()
    {
        using S_ = typename Problem::BlockShape;
        if constexpr(std::is_same_v<typename Problem::ADataType, ck_tile::bf16_t> &&
                     std::is_same_v<typename Problem::GDataType, ck_tile::bf16_t> &&
                     S_::Block_M0 == 32 && S_::Block_N0 == 512 && S_::Block_K0 == 128 &&
                     S_::Warp_M0 == 16 && S_::Warp_N0 == 16 && S_::Warp_K0 == 32)
        {
            return Flatmm_32x512x128_1x4x1_16x16x32_BF16{};
        }
        else if constexpr(std::is_same_v<typename Problem::ADataType, ck_tile::fp16_t> &&
                          std::is_same_v<typename Problem::GDataType, ck_tile::fp16_t> &&
                          S_::Block_M0 == 32 && S_::Block_N0 == 512 && S_::Block_K0 == 128 &&
                          S_::Warp_M0 == 16 && S_::Warp_N0 == 16 && S_::Warp_K0 == 32)
        {
            return Flatmm_32x512x128_1x4x1_16x16x32_FP16{};
        }
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetUK_1()
    {
        using S_ = typename Problem::BlockShape;
        using T_ = typename Problem::Traits;
        if constexpr(std::is_same_v<typename Problem::YDataType, ck_tile::bf16_t> &&
                     std::is_same_v<typename Problem::DDataType, ck_tile::bf16_t> &&
                     std::is_same_v<typename Problem::TopkWeightDataType, float> &&
                     S_::Block_M1 == 32 && S_::Block_N1 == 128 && S_::Block_K1 == 512 &&
                     S_::Warp_M0 == 16 && S_::Warp_N0 == 16 && S_::Warp_K0 == 32 &&
                     T_::PipeInterleave == false)
        {
            return FlatmmSn_32x128x512_1x4x1_16x16x32_BF16{};
            // return FlatmmSn_32x128x512_1x4x1_16x16x32_BF16_itl{};
        }
        else if constexpr(std::is_same_v<typename Problem::YDataType, ck_tile::fp16_t> &&
                          std::is_same_v<typename Problem::DDataType, ck_tile::fp16_t> &&
                          std::is_same_v<typename Problem::TopkWeightDataType, float> &&
                          S_::Block_M1 == 32 && S_::Block_N1 == 128 && S_::Block_K1 == 512 &&
                          S_::Warp_M0 == 16 && S_::Warp_N0 == 16 && S_::Warp_K0 == 32 &&
                          T_::PipeInterleave == false)
        {
            return FlatmmSn_32x128x512_1x4x1_16x16x32_FP16{};
            // return FlatmmSn_32x128x512_1x4x1_16x16x32_FP16_itl{};
        }
        else if constexpr(std::is_same_v<typename Problem::YDataType, ck_tile::bf16_t> &&
                          std::is_same_v<typename Problem::DDataType, ck_tile::bf16_t> &&
                          std::is_same_v<typename Problem::TopkWeightDataType, float> &&
                          S_::Block_M1 == 32 && S_::Block_N1 == 128 && S_::Block_K1 == 512 &&
                          S_::Warp_M0 == 16 && S_::Warp_N0 == 16 && S_::Warp_K0 == 32 &&
                          T_::PipeInterleave == true)
        {
            // return FlatmmSn_32x128x512_1x4x1_16x16x32_FP16{};
            return FlatmmSn_32x128x512_1x4x1_16x16x32_BF16_itl{};
        }
        else if constexpr(std::is_same_v<typename Problem::YDataType, ck_tile::fp16_t> &&
                          std::is_same_v<typename Problem::DDataType, ck_tile::fp16_t> &&
                          std::is_same_v<typename Problem::TopkWeightDataType, float> &&
                          S_::Block_M1 == 32 && S_::Block_N1 == 128 && S_::Block_K1 == 512 &&
                          S_::Warp_M0 == 16 && S_::Warp_N0 == 16 && S_::Warp_K0 == 32 &&
                          T_::PipeInterleave == true)
        {
            // return FlatmmSn_32x128x512_1x4x1_16x16x32_FP16{};
            return FlatmmSn_32x128x512_1x4x1_16x16x32_FP16_itl{};
        }
    }
};
} // namespace ck_tile
