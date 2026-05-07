// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_whole_k_prefetch_default_policy.hpp"
#include "ck_tile/ops/fmha/block/block_dropout.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

template <typename Problem_, typename Policy_ = BlockFmhaPipelineQRKSVSWholeKPrefetchDefaultPolicy>
struct BlockFmhaPipelineQRKSVSWholeKPrefetch
{
    using Problem               = remove_cvref_t<Problem_>;
    using Policy                = remove_cvref_t<Policy_>;
    using QDataType             = remove_cvref_t<typename Problem::QDataType>;
    using KDataType             = remove_cvref_t<typename Problem::KDataType>;
    using VDataType             = remove_cvref_t<typename Problem::VDataType>;
    using SaccDataType          = remove_cvref_t<typename Problem::SaccDataType>;
    using SMPLComputeDataType   = remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using BiasDataType          = remove_cvref_t<typename Problem::BiasDataType>;
    using RandValOutputDataType = remove_cvref_t<typename Problem::RandValOutputDataType>;
    using LSEDataType           = remove_cvref_t<typename Problem::LSEDataType>;
    using PDataType             = remove_cvref_t<typename Problem::PDataType>;
    using OaccDataType          = remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType             = remove_cvref_t<typename Problem::ODataType>;
    using FmhaMask              = remove_cvref_t<typename Problem::FmhaMask>;
    using AttentionVariant      = remove_cvref_t<typename Problem::AttentionVariant>;

    using BlockFmhaShape             = remove_cvref_t<typename Problem::BlockFmhaShape>;
    using VLayout                    = remove_cvref_t<typename BlockFmhaShape::VLayout>;
    static constexpr bool kQLoadOnce = true;
    static_assert(kQLoadOnce == Policy::QLoadOnce);

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0           = BlockFmhaShape::kM0;
    static constexpr index_t kN0           = BlockFmhaShape::kN0;
    static constexpr index_t kK0           = BlockFmhaShape::kK0;
    static constexpr index_t kN1           = BlockFmhaShape::kN1;
    static constexpr index_t kK1           = BlockFmhaShape::kK1;
    static constexpr index_t kQKHeaddim    = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kSubQKHeaddim = BlockFmhaShape::kSubQKHeaddim;

    static_assert(kSubQKHeaddim <= 256, "hdim bigger than 256 is not suitable for this pipeline!");

    static constexpr bool kIsGroupMode = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ  = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK  = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV = (kQKHeaddim < kSubQKHeaddim) ? 1 : Problem::kPadHeadDimV;
    static constexpr auto BiasEnum     = Problem::BiasEnum;
    static constexpr bool kStoreLSE    = Problem::kStoreLSE;
    static constexpr bool kHasDropout  = Problem::kHasDropout;
    static constexpr bool kHasLogitsSoftCap = Problem::kHasLogitsSoftCap;

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV = []() {
        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            return Problem::kPadHeadDimV ? 1 : Policy::template GetAlignmentV<Problem>();
        else
            return kPadSeqLenK ? 1 : Policy::template GetAlignmentV<Problem>();
    }();

    static constexpr index_t kAlignmentO =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentO<Problem>();
    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();
    static constexpr index_t kAlignmentRandVal =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentRandVal<Problem>();

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            if constexpr(kQKHeaddim == 32)
            {
                return 2;
            }
            else if constexpr(kQKHeaddim == 64)
            {
                return 2;
            }
            else if constexpr(kQKHeaddim == 96 || kQKHeaddim == 128)
            {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                    return 1;
                else
                    return 2;
            }
            else if constexpr(kQKHeaddim == 256)
            {
                return 1;
            }
            else
            {
                return 1;
            };
        }
    }();

    static constexpr const char* name = "qr_async";

    using DropoutType = std::conditional_t<kHasDropout, BlockDropout, NullBlockDropout>;

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename VElementFunction,
              typename BiasElementFunction,
              typename LSEElementFunction,
              typename SAccElementFunction,
              typename PComputeElementFunction,
              typename OAccElementFunction,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*kSubQKHeaddim tile
               const QElementFunction& q_element_func,
               const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*kSubQKHeaddim tile
               const KElementFunction& k_element_func,
               const VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
               const VElementFunction& v_element_func,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               const BiasElementFunction& bias_element_func,
               RandValDramBlockWindowTmp& randval_dram_block_window_tmp,
               LSEDramBlockWindowTmp& lse_dram_window_tmp, // M0*1 tile
               const LSEElementFunction& lse_element_func,
               const SAccElementFunction& s_acc_element_func,
               const PComputeElementFunction& p_compute_element_func,
               const OAccElementFunction& o_acc_element_func,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& /* unused */,
               const AttentionVariantParams& /* unused */,
               const BlockIndices& /* unused */,
               void* smem_ptr,
               DropoutType& dropout) const
    {
        ignore = q_element_func;
        ignore = k_element_func;

        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK0 == KDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kN1 == VDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kK1 == VDramBlockWindowTmp{}.get_window_lengths()[number<1>{}] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        constexpr auto I0 = number<0>{};
        constexpr auto I1 = number<1>{};

        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;
        static_assert(2 <= k0_loops);
        static_assert(2 <= k1_loops);

        constexpr bool kPreloadWholeNextIterationK =
            Policy::template IsPreloadWholeNextIterationK<Problem>();

        constexpr auto NumKLdsBuffers = Policy::template GetNumKLdsBuffers<Problem>();
        constexpr auto NumVLdsBuffers = Policy::template GetNumVLdsBuffers<Problem>();
        constexpr auto NumPrefetchV   = Policy::template GetNumPrefetchV<Problem>();

        static_assert(NumKLdsBuffers >= 2);

        auto q_dram_window = make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                                              q_dram_block_window_tmp.get_window_lengths(),
                                              q_dram_block_window_tmp.get_window_origin(),
                                              Policy::template MakeQRegTileDistribution<Problem>());

        const auto q_origin = q_dram_window.get_window_origin();
        const auto [seqlen_k_start, seqlen_k_end] =
            mask.GetTileRangeAlongX(q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{});

        auto k_dram_block_window =
            make_tile_window(k_dram_block_window_tmp.get_bottom_tensor_view(),
                             k_dram_block_window_tmp.get_window_lengths(),
                             {seqlen_k_start, 0});

        auto k_dram_window =
            make_tile_window(k_dram_block_window.get_bottom_tensor_view(),
                             k_dram_block_window.get_window_lengths(),
                             k_dram_block_window.get_window_origin(),
                             Policy::template MakeKDramTileDistribution<Problem>());

        using k_tile_type = decltype(load_tile(k_dram_window));

        auto k_tiles = [&]() {
            if constexpr(kPreloadWholeNextIterationK)
                return statically_indexed_array<k_tile_type, k0_loops>{};
            else
                return statically_indexed_array<k_tile_type, 1>{};
        }();

        k_tiles[I0] = load_tile(k_dram_window);
        move_tile_window(k_dram_window, {0, kK0});

        auto q_tile = load_tile(q_dram_window);

        __builtin_amdgcn_sched_barrier(0);

        // K tile in LDS
        KDataType* k_lds_ptr = static_cast<KDataType*>(smem_ptr);
        auto k_lds           = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsBlockDescriptor<Problem>());
        auto k_lds_window = make_tile_window(
            k_lds, Policy::template MakeKLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        using k_lds_window_type =
            decltype(get_slice_tile(k_lds_window, sequence<0, 0>{}, sequence<kN0, kK0>{}));

        statically_indexed_array<k_lds_window_type, NumKLdsBuffers> k_lds_windows;

        static_for<0, NumKLdsBuffers, 1>{}([&](auto i_buf) {
            k_lds_windows[i_buf] = get_slice_tile(
                k_lds_window, sequence<i_buf * kN0, 0>{}, sequence<(i_buf + 1) * kN0, kK0>{});
        });

        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.get_bottom_tensor_view(),
                             v_dram_block_window_tmp.get_window_lengths(),
                             {0, seqlen_k_start}, // TODO: hdim split?
                             Policy::template MakeVDramTileDistribution<Problem>());
        // V tile in LDS
        auto v_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType*>(static_cast<char*>(smem_ptr) +
                                         Policy::template GetExclusiveKLdsBytes<Problem>()),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_window = make_tile_window(
            v_lds, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        using v_tile_type = decltype(load_tile(v_dram_window));

        statically_indexed_array<v_tile_type, NumPrefetchV> v_tiles;

        using v_lds_window_type =
            decltype(get_slice_tile(v_lds_window, sequence<0, 0>{}, sequence<kN1, kK1>{}));

        statically_indexed_array<v_lds_window_type, NumVLdsBuffers> v_lds_windows;

        static_for<0, NumVLdsBuffers, 1>{}([&](auto i_buf) {
            v_lds_windows[i_buf] = get_slice_tile(
                v_lds_window, sequence<i_buf * kN1, 0>{}, sequence<(i_buf + 1) * kN1, kK1>{});
        });

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetKVBlockGemm<Problem>();

        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        auto s_acc              = SaccBlockTileType{};

        // reduction function for softmax
        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType = decltype(cast_tile<SMPLComputeDataType>(s_acc));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());

        // init Oacc, M, L
        auto o_acc = OaccBlockTileType{};
        auto m     = MLBlockTileType{};
        auto l     = MLBlockTileType{};

        clear_tile(o_acc);
        set_tile(m, -numeric<SMPLComputeDataType>::infinity());
        clear_tile(l);

        const auto num_total_loop = integer_divide_ceil(seqlen_k_end - seqlen_k_start, kN0);

        // check early exit if no work to do
        if constexpr(FmhaMask::IsMasking || kPadSeqLenK)
        {
            if(num_total_loop <= 0)
            {
                if constexpr(kStoreLSE)
                {
                    auto lse =
                        make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

                    set_tile(lse, -numeric<SMPLComputeDataType>::infinity());

                    store_tile(lse_dram_window_tmp, tile_elementwise_in(lse_element_func, lse));
                }

                // Note: here occ are all cleard, return it
                // Note: q loaded but no fence, ignore it.
                return o_acc;
            }
        }

        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {bias_origin.at(number<0>{}), seqlen_k_start}, // M/N
                             Policy::template MakeBiasDramTileDistribution<decltype(gemm_0)>());

        auto randval_dram_window = dropout.template MakeRandvalDramWindow<decltype(gemm_0)>(
            randval_dram_block_window_tmp, seqlen_k_start);

        q_tile = tile_elementwise_in(q_element_func, q_tile);

        index_t i_total_loops = 0;

        do
        {
            if constexpr(kPreloadWholeNextIterationK)
            {
                if(i_total_loops == 0) // executed by fist iteration
                {
                    if(num_total_loop > 1) // there are multiple iterations
                    {
                        static_for<0, k0_loops - 1, 1>{}([&](auto i_k0) {
                            store_tile(
                                k_lds_windows[number<i_k0 % NumKLdsBuffers>{}],
                                tile_elementwise_in(k_element_func, k_tiles[number<i_k0>{}]));

                            k_tiles[number<i_k0 + 1>{}] = load_tile(k_dram_window);
                            if constexpr(i_k0 < k0_loops - 2)
                                move_tile_window(k_dram_window, {0, kK0});

                            if constexpr(i_k0 == 0)
                                clear_tile(s_acc);

                            block_sync_lds();
                            // execute current unroll of gemm_0
                            gemm_0(s_acc,
                                   get_slice_tile(q_tile,
                                                  sequence<0, i_k0 * kK0>{},
                                                  sequence<kM0, (i_k0 + 1) * kK0>{}),
                                   k_lds_windows[number<i_k0 % NumKLdsBuffers>{}]);
                        });

                        store_tile(
                            k_lds_windows[number<(k0_loops - 1) % NumKLdsBuffers>{}],
                            tile_elementwise_in(k_element_func, k_tiles[number<k0_loops - 1>{}]));

                        // prefetch first v_tile
                        v_tiles[I0] = load_tile(v_dram_window);
                        move_tile_window(v_dram_window, {0, kK1});

                        move_tile_window(k_dram_window, {kN0, -(k0_loops - 1) * kK0});

                        // prefetch all k_tiles for next iteration
                        static_for<0, k0_loops, 1>{}([&](auto i_k0) {
                            k_tiles[number<i_k0>{}] = load_tile(k_dram_window);

                            if constexpr(i_k0 < k0_loops - 1)
                                move_tile_window(k_dram_window, {0, kK0});
                        });

                        move_tile_window(k_dram_window, {0, -(k0_loops - 1) * kK0});

                        block_sync_lds();
                        // execute last unroll of gemm_0
                        gemm_0(s_acc,
                               get_slice_tile(q_tile,
                                              sequence<0, (k0_loops - 1) * kK0>{},
                                              sequence<kM0, k0_loops * kK0>{}),
                               k_lds_windows[number<(k0_loops - 1) % NumKLdsBuffers>{}]);
                    }
                    else // there is only single iteration
                    {
                        static_for<0, k0_loops - 1, 1>{}([&](auto i_k0) {
                            store_tile(
                                k_lds_windows[number<i_k0 % NumKLdsBuffers>{}],
                                tile_elementwise_in(k_element_func, k_tiles[number<i_k0>{}]));

                            k_tiles[number<i_k0 + 1>{}] = load_tile(k_dram_window);
                            if constexpr(i_k0 < k0_loops - 2)
                                move_tile_window(k_dram_window, {0, kK0});

                            if constexpr(i_k0 == 0)
                                clear_tile(s_acc);

                            block_sync_lds();
                            // execute current unroll of gemm_0
                            gemm_0(s_acc,
                                   get_slice_tile(q_tile,
                                                  sequence<0, i_k0 * kK0>{},
                                                  sequence<kM0, (i_k0 + 1) * kK0>{}),
                                   k_lds_windows[number<i_k0 % NumKLdsBuffers>{}]);
                        });

                        store_tile(
                            k_lds_windows[number<(k0_loops - 1) % NumKLdsBuffers>{}],
                            tile_elementwise_in(k_element_func, k_tiles[number<k0_loops - 1>{}]));

                        // prefetch first v_tile
                        v_tiles[I0] = load_tile(v_dram_window);
                        move_tile_window(v_dram_window, {0, kK1});

                        block_sync_lds();
                        gemm_0(s_acc,
                               get_slice_tile(q_tile,
                                              sequence<0, (k0_loops - 1) * kK0>{},
                                              sequence<kM0, k0_loops * kK0>{}),
                               k_lds_windows[number<(k0_loops - 1) % NumKLdsBuffers>{}]);

                        // move_tile_window(k_dram_window, {0, -k0_loops * kK0});
                    }
                }
                else // executed by intermediate and last iteration
                {
                    if(i_total_loops < num_total_loop - 1) // intermediate iteration
                    {
                        store_tile(k_lds_windows[I0],
                                   tile_elementwise_in(k_element_func, k_tiles[I0]));

                        // prefetch first v_tile
                        v_tiles[I0] = load_tile(v_dram_window);
                        move_tile_window(v_dram_window, {0, kK1});

                        clear_tile(s_acc);
                        block_sync_lds();
                        gemm_0(s_acc,
                               get_slice_tile(q_tile, sequence<0, 0>{}, sequence<kM0, kK0>{}),
                               k_lds_windows[I0]);

                        store_tile(k_lds_windows[I1],
                                   tile_elementwise_in(k_element_func, k_tiles[I1]));

                        move_tile_window(k_dram_window, {kN0, 0});

                        // prefetch first k_tile for next iteration
                        k_tiles[I0] = load_tile(k_dram_window);
                        move_tile_window(k_dram_window, {0, kK0});

                        k_tiles[I1] = load_tile(k_dram_window);
                        if constexpr(1 < k0_loops - 1)
                            move_tile_window(k_dram_window, {0, kK0});

                        block_sync_lds();
                        gemm_0(s_acc,
                               get_slice_tile(q_tile, sequence<0, kK0>{}, sequence<kM0, 2 * kK0>{}),
                               k_lds_windows[I1]);

                        // during the gemm-loop, also prefetch other k_tiles for next iteration
                        static_for<2, k0_loops, 1>{}([&](auto i_k0) {
                            store_tile(k_lds_windows[number<i_k0 % NumKLdsBuffers>{}],
                                       k_tiles[number<i_k0>{}]);

                            k_tiles[number<i_k0>{}] = load_tile(k_dram_window);
                            if constexpr(i_k0 < k0_loops - 1)
                                move_tile_window(k_dram_window, {0, kK0});

                            block_sync_lds();
                            gemm_0(s_acc,
                                   get_slice_tile(q_tile,
                                                  sequence<0, i_k0 * kK0>{},
                                                  sequence<kM0, (i_k0 + 1) * kK0>{}),
                                   k_lds_windows[number<i_k0 % NumKLdsBuffers>{}]);
                        });

                        move_tile_window(k_dram_window, {0, -(k0_loops - 1) * kK0});
                    }
                    else // last iteration
                    {
                        store_tile(k_lds_windows[I0],
                                   tile_elementwise_in(k_element_func, k_tiles[I0]));

                        // prefetch first v_tile
                        v_tiles[I0] = load_tile(v_dram_window);
                        move_tile_window(v_dram_window, {0, kK1});

                        clear_tile(s_acc);
                        block_sync_lds();
                        gemm_0(s_acc,
                               get_slice_tile(q_tile, sequence<0, 0>{}, sequence<kM0, kK0>{}),
                               k_lds_windows[I0]);

                        static_for<1, k0_loops, 1>{}([&](auto i_k0) {
                            store_tile(
                                k_lds_windows[number<i_k0 % NumKLdsBuffers>{}],
                                tile_elementwise_in(k_element_func, k_tiles[number<i_k0>{}]));

                            block_sync_lds();
                            gemm_0(s_acc,
                                   get_slice_tile(q_tile,
                                                  sequence<0, i_k0 * kK0>{},
                                                  sequence<kM0, (i_k0 + 1) * kK0>{}),
                                   k_lds_windows[number<i_k0 % NumKLdsBuffers>{}]);
                        });
                    };
                };
            }
            else // only preload one unroll of K for next iteration
            {
                static_for<0, k0_loops - 1, 1>{}([&](auto i_k0) {
                    store_tile(k_lds_windows[number<i_k0 % NumKLdsBuffers>{}],
                               tile_elementwise_in(k_element_func, k_tiles[I0]));
                    if constexpr(i_k0 == 0)
                        clear_tile(s_acc);

                    if constexpr(i_k0 < k0_loops - 1)
                        k_tiles[I0] = load_tile(k_dram_window);
                    if constexpr(i_k0 < k0_loops - 2)
                        move_tile_window(k_dram_window, {0, kK0});

                    block_sync_lds();
                    // execute current unroll of gemm_0
                    gemm_0(s_acc,
                           get_slice_tile(q_tile,
                                          sequence<0, i_k0 * kK0>{},
                                          sequence<kM0, (i_k0 + 1) * kK0>{}),
                           k_lds_windows[number<i_k0 % NumKLdsBuffers>{}]);
                });

                store_tile(k_lds_windows[number<(k0_loops - 1) % NumKLdsBuffers>{}],
                           tile_elementwise_in(k_element_func, k_tiles[I0]));

                // prefetch first v_tile
                v_tiles[I0] = load_tile(v_dram_window);
                move_tile_window(v_dram_window, {0, kK1});

                block_sync_lds();
                gemm_0(s_acc,
                       get_slice_tile(q_tile,
                                      sequence<0, (k0_loops - 1) * kK0>{},
                                      sequence<kM0, k0_loops * kK0>{}),
                       k_lds_windows[number<(k0_loops - 1) % NumKLdsBuffers>{}]);
            };

            __builtin_amdgcn_sched_barrier(0);

            const auto bias_tile = load_tile(bias_dram_window); // load bias tile

            static_for<1, NumPrefetchV, 1>{}([&](auto i_buf) {
                v_tiles[i_buf] = load_tile(v_dram_window);
                move_tile_window(v_dram_window, {0, kK1});
            });

            // STAGE 2, scale_s, add bias, mask, softmax
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                s_acc = tile_elementwise_in(s_acc_element_func, s_acc);
                tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
                tile_elementwise_inout(
                    [&](auto& x, const auto& y) {
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                        x += type_convert<SaccDataType>(bias_element_func(y));
#else
                        x += log2e_v<SaccDataType> *
                             type_convert<SaccDataType>(bias_element_func(y));
#endif
                    },
                    s_acc,
                    bias_tile);
            }
            else if constexpr(BiasEnum == BlockAttentionBiasEnum::ALIBI)
            {
                const auto k_origin    = k_dram_block_window.get_window_origin();
                constexpr auto s_spans = decltype(s_acc)::get_distributed_spans();
                s_acc                  = tile_elementwise_in(s_acc_element_func, s_acc);
                sweep_tile_span(s_spans[number<0>{}], [&](auto idx0) {
                    sweep_tile_span(s_spans[number<1>{}], [&](auto idx1) {
                        const auto tile_idx = get_x_indices_from_distributed_indices(
                            s_acc.get_tile_distribution(), make_tuple(idx0, idx1));

                        const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                        const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                        constexpr auto i_j_idx = make_tuple(idx0, idx1);

                        s_acc(i_j_idx) *= scale_s;
                        position_encoding.update(s_acc(i_j_idx), row, col);
                    });
                });
            }
            else
            {
                s_acc = tile_elementwise_in(s_acc_element_func, s_acc);
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
#endif
            }
            move_tile_window(bias_dram_window, {0, kN0});
            if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
            {
                const auto k_origin      = k_dram_block_window.get_window_origin();
                bool need_perpixel_check = mask.IsEdgeTile(q_origin.at(number<0>{}),
                                                           k_origin.at(number<0>{}),
                                                           number<kM0>{},
                                                           number<kN0>{});
                if(need_perpixel_check)
                {
                    set_tile_if(
                        s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                            const auto row = q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                            const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                            return mask.IsOutOfBound(row, col);
                        });
                }
            }

            const auto s = cast_tile<SMPLComputeDataType>(s_acc); // S{j}
            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                s,
                sequence<1>{},
                f_max,
                -numeric<SMPLComputeDataType>::infinity()); // m_local = rowmax(S{j})
            block_tile_reduce_sync(m_local, f_max, bool_constant<false>{});

            const auto m_old = m; // m{j-1}
            tile_elementwise_inout(
                [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local); // m{j}

            auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                s.get_tile_distribution()); // Pcompute{j}

            static const auto get_validated_m = [](SMPLComputeDataType raw_m) {
                /// NOTICE: bias might be materialized mask including -inf values, need
                /// consideration
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             FmhaMask::IsMasking)
                {
                    return raw_m == -numeric<SMPLComputeDataType>::infinity()
                               ? type_convert<SMPLComputeDataType>(0.f)
                               : raw_m;
                }
                else
                {
                    return raw_m;
                }
            };

            constexpr auto p_spans = decltype(p_compute)::get_distributed_spans();
            sweep_tile_span(p_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                auto row_max = scale_s * get_validated_m(m[i_idx]);
#endif
                sweep_tile_span(p_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        p_compute(i_j_idx) = exp2(s[i_j_idx] - get_validated_m(m[i_idx]));
                    }
                    else
                    {
                        p_compute(i_j_idx) = exp2(scale_s * s[i_j_idx] - row_max);
                    }
#else
                    p_compute(i_j_idx)     = exp(s[i_j_idx] - get_validated_m(m[i_idx]));
#endif
                });
            });

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0}); // rowsum(Pcompute{j})

            block_tile_reduce_sync(rowsum_p, f_sum, bool_constant<false>{});
            // l{j}, Oacc{j}
            constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();
            sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                const auto tmp = [&]() {
                    if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                                 BiasEnum == BlockAttentionBiasEnum::ALIBI)
                    {
                        return exp2(m_old[i_idx] - get_validated_m(m[i_idx]));
                    }
                    else
                    {
                        auto row_max = scale_s * get_validated_m(m[i_idx]);
                        return exp2(scale_s * m_old[i_idx] - row_max);
                    }
                }();
#else
                    const auto tmp = exp(m_old[i_idx] - get_validated_m(m[i_idx]));
#endif
                l(i_idx) = tmp * l[i_idx] + rowsum_p[i_idx];
                sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    // FIXME: this use different equation from FA v2 paper,
                    // but produce correc result.
                    // Is the equation wrong?
                    o_acc(i_j_idx) *= tmp;
                });
            });

            if constexpr(kHasDropout)
            {
                auto randval_ptr =
                    reinterpret_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>();
                dropout.template Run<decltype(gemm_0), SMPLComputeDataType, RandValOutputDataType>(
                    smem_ptr, seqlen_k_start + i_total_loops * kN0, p_compute, randval_dram_window);
            }

            __builtin_amdgcn_sched_barrier(0x7f);

            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                    Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                shuffle_tile(v_shuffle_tmp, v_tiles[I0]);

                store_tile(
                    v_lds_windows[I0],
                    tile_elementwise_in(v_element_func, v_shuffle_tmp)); // store the prefetch
            }
            else
            {
                store_tile(v_lds_windows[I0],
                           tile_elementwise_in(v_element_func, v_tiles[I0])); // store the prefetch
            }

            __builtin_amdgcn_sched_barrier(0);

            const auto p =
                cast_tile<PDataType>(tile_elementwise_in(p_compute_element_func, p_compute));

            if constexpr(!kPreloadWholeNextIterationK)
            {
                if(i_total_loops < num_total_loop - 1)
                {
                    move_tile_window(k_dram_window, {kN0, -(k0_loops - 1) * kK0});
                    k_tiles[I0] = load_tile(k_dram_window);
                    move_tile_window(k_dram_window, {0, kK0});
                };

                __builtin_amdgcn_sched_barrier(0);
            }

            // STAGE 3, KV gemm
            if constexpr(k1_loops > 1)
            {
                if constexpr(NumPrefetchV == 1) // NumVLdsBuffers == 2
                {
                    static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                        v_tiles[I0] = load_tile(v_dram_window);

                        block_sync_lds();
                        gemm_1(o_acc,
                               get_slice_tile(
                                   p, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{}),
                               v_lds_windows[number<i_k1 % NumVLdsBuffers>{}]);

                        if constexpr(std::is_same_v<VLayout,
                                                    ck_tile::tensor_layout::gemm::RowMajor>)
                        {
                            auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                                Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                            shuffle_tile(v_shuffle_tmp, v_tiles[I0]);
                            store_tile(v_lds_windows[number<(i_k1 + 1) % NumVLdsBuffers>{}],
                                       tile_elementwise_in(v_element_func, v_shuffle_tmp));
                        }
                        else
                        {
                            store_tile(v_lds_windows[number<(i_k1 + 1) % NumVLdsBuffers>{}],
                                       tile_elementwise_in(v_element_func, v_tiles[I0]));
                        }

                        move_tile_window(v_dram_window, {0, kK1});
                    });
                }
                else // NumVLdsBuffers == 3 or 2
                {
                    static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                        if constexpr(i_k1 < k1_loops - NumPrefetchV)
                            v_tiles[number<i_k1 % NumPrefetchV>{}] = load_tile(v_dram_window);

                        block_sync_lds();
                        gemm_1(o_acc,
                               get_slice_tile(
                                   p, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{}),
                               v_lds_windows[number<i_k1 % NumVLdsBuffers>{}]);

                        if constexpr(std::is_same_v<VLayout,
                                                    ck_tile::tensor_layout::gemm::RowMajor>)
                        {
                            auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                                Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                            shuffle_tile(v_shuffle_tmp,
                                         v_tiles[number<(i_k1 + 1) % NumPrefetchV>{}]);
                            store_tile(v_lds_windows[number<(i_k1 + 1) % NumVLdsBuffers>{}],
                                       tile_elementwise_in(v_element_func, v_shuffle_tmp));
                        }
                        else
                        {
                            store_tile(
                                v_lds_windows[number<(i_k1 + 1) % NumVLdsBuffers>{}],
                                tile_elementwise_in(v_element_func,
                                                    v_tiles[number<(i_k1 + 1) % NumPrefetchV>{}]));
                        }

                        if constexpr(i_k1 < k1_loops - NumPrefetchV)
                            move_tile_window(v_dram_window, {0, kK1});
                    });
                }
            }
            // move K tile windows
            move_tile_window(k_dram_block_window, {kN0, 0});

            block_sync_lds();
            gemm_1(o_acc,
                   get_slice_tile(p, sequence<0, (k1_loops - 1) * kK1>{}, sequence<kM0, kN0>{}),
                   v_lds_windows[number<(k1_loops - 1) % NumVLdsBuffers>{}]);

            if constexpr(Policy::template IsFirstKLdsBufferOverlapLastVLdsBuffer<Problem>())
            {
                __builtin_amdgcn_sched_barrier(0);
                __builtin_amdgcn_s_barrier();
            };

        } while(++i_total_loops < num_total_loop);

        // store lse
        if constexpr(kStoreLSE)
        {
            auto lse = make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

            constexpr auto lse_spans = decltype(lse)::get_distributed_spans();
            sweep_tile_span(lse_spans[number<0>{}], [&, m_ = m, l_ = l](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             BiasEnum == BlockAttentionBiasEnum::ALIBI)
                {
                    lse(i_idx) = m_[i_idx] / C_LOG2E + log(l_[i_idx]);
                }
                else
                {
                    lse(i_idx) = m_[i_idx] * scale_s / C_LOG2E + log(l_[i_idx]);
                }
#else
                lse(i_idx) = m_[i_idx] + log(l_[i_idx]);
#endif
            });

            store_tile(lse_dram_window_tmp, tile_elementwise_in(lse_element_func, lse));
        }

        // finally, O
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const auto tmp       = [&]() {
                if constexpr(FmhaMask::IsMasking)
                {
                    return l[i_idx] == 0.f ? 0.f : 1 / l[i_idx];
                }
                else
                    return 1 / l[i_idx];
            }();
            sweep_tile_span(o_spans[number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });

        o_acc = tile_elementwise_in(o_acc_element_func, o_acc);

        return o_acc;
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename RandValDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,       // M0*K0 tile
               const KDramBlockWindowTmp& k_dram_block_window_tmp,       // N0*K0 tile
               const VDramBlockWindowTmp& v_dram_block_window_tmp,       // N1*K1 tile
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               RandValDramBlockWindowTmp& randval_dram_block_window_tmp, // M0*N0 tile
               LSEDramBlockWindowTmp& lse_dram_block_window_tmp,         // M0*1 tile
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               void* smem_ptr,
               DropoutType& dropout) const
    {
        return operator()(q_dram_block_window_tmp,
                          identity{},
                          k_dram_block_window_tmp,
                          identity{},
                          v_dram_block_window_tmp,
                          identity{},
                          bias_dram_block_window_tmp,
                          identity{},
                          randval_dram_block_window_tmp,
                          lse_dram_block_window_tmp,
                          identity{},
                          identity{},
                          identity{},
                          identity{},
                          mask,
                          position_encoding,
                          scale_s,
                          variant,
                          variant_params,
                          block_indices,
                          smem_ptr,
                          dropout);
    }
};

} // namespace ck_tile
