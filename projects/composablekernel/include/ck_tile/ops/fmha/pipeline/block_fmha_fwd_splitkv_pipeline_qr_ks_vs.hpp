// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/block_attention_bias_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs_default_policy.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"

namespace ck_tile {

// This pipeline is qkv all located in LDS
template <typename Problem_, typename Policy_ = BlockFmhaFwdSplitKVPipelineQRKSVSDefaultPolicy>
struct BlockFmhaFwdSplitKVPipelineQRKSVS
{
    using Problem             = remove_cvref_t<Problem_>;
    using Policy              = remove_cvref_t<Policy_>;
    using QDataType           = remove_cvref_t<typename Problem::QDataType>;
    using KDataType           = remove_cvref_t<typename Problem::KDataType>;
    using VDataType           = remove_cvref_t<typename Problem::VDataType>;
    using SaccDataType        = remove_cvref_t<typename Problem::SaccDataType>;
    using SMPLComputeDataType = remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using BiasDataType        = remove_cvref_t<typename Problem::BiasDataType>;
    using LSEDataType         = remove_cvref_t<typename Problem::LSEDataType>;
    using PDataType           = remove_cvref_t<typename Problem::PDataType>;
    using OaccDataType        = remove_cvref_t<typename Problem::OaccDataType>;
    using ODataType           = remove_cvref_t<typename Problem::ODataType>;
    using AttentionVariant    = remove_cvref_t<typename Problem::AttentionVariant>;
    using FmhaMask            = remove_cvref_t<typename Problem::FmhaMask>;

    using BlockFmhaShape             = remove_cvref_t<typename Problem::BlockFmhaShape>;
    using VLayout                    = remove_cvref_t<typename BlockFmhaShape::VLayout>;
    static constexpr bool kQLoadOnce = true; // if q_tile load whole block length (hdim) at once
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

    static constexpr bool kIsGroupMode      = Problem::kIsGroupMode;
    static constexpr bool kPadSeqLenQ       = Problem::kPadSeqLenQ;
    static constexpr bool kPadSeqLenK       = Problem::kPadSeqLenK;
    static constexpr bool kPadHeadDimQ      = Problem::kPadHeadDimQ;
    static constexpr bool kPadHeadDimV      = Problem::kPadHeadDimV;
    static constexpr bool kHasLogitsSoftCap = Problem::kHasLogitsSoftCap;
    static constexpr auto BiasEnum          = Problem::BiasEnum;
    static constexpr bool kStoreLSE         = Problem::kStoreLSE;
    static constexpr bool kIsPagedKV        = Problem::kIsPagedKV;
    static constexpr bool kHasUnevenSplits  = Problem::kHasUnevenSplits;
    static constexpr bool kHasSink          = Problem::kHasSink;

    static_assert((CK_TILE_FMHA_FWD_FAST_EXP2 &&
                   (kHasLogitsSoftCap && Problem::BiasEnum == BlockAttentionBiasEnum::NO_BIAS ||
                    !kHasLogitsSoftCap)) ||
                  (!CK_TILE_FMHA_FWD_FAST_EXP2 && !kHasLogitsSoftCap));

    // last dimension vector length used to create tensor view(and decide buffer_load vector length)
    // ... together with tensor distribution. tensor dist should able to overwrite this
    static constexpr index_t kAlignmentQ =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentQ<Problem>();
    static constexpr index_t kAlignmentK =
        kPadHeadDimQ ? 1 : Policy::template GetAlignmentK<Problem>();
    static constexpr index_t kAlignmentV = []() {
        if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            return kPadHeadDimV ? 1 : Policy::template GetAlignmentV<Problem>();
        else
            return kPadSeqLenK ? 1 : Policy::template GetAlignmentV<Problem>();
    }();

    static constexpr index_t kAlignmentOacc =
        kPadHeadDimV ? 1 : Policy::template GetAlignmentOacc<Problem>();

    static constexpr index_t kAlignmentBias =
        kPadSeqLenK ? 1 : Policy::template GetAlignmentBias<Problem>();

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            if constexpr(kQKHeaddim <= 32)
            {
                return 2;
            }
            else if constexpr(kQKHeaddim <= 64)
            {
                return 3;
            }
            else if constexpr(kQKHeaddim <= 128)
            {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
                    return 1;
                else
                    return 2;
            }
            else if constexpr(kQKHeaddim <= 256)
            {
                return 1;
            }
            else
            {
                return 1;
            }
        }
    }();

    static constexpr const char* name = "qr";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowLengths,
              typename KPageBlockNavigator,
              typename VDramBlockWindowLengths,
              typename VPageBlockNavigator,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename QElementFunction,
              typename KElementFunction,
              typename VElementFunction,
              typename BiasElementFunction,
              typename LSEaccElementFunction,
              typename SAccElementFunction,
              typename PComputeElementFunction,
              typename OAccElementFunction,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*K0 tile
               const QElementFunction& q_element_func,
               const KDramBlockWindowLengths& k_dram_block_window_lengths, // N0*K0 tile
               const KPageBlockNavigator& k_page_block_navigator,
               const KElementFunction& k_element_func,
               const VDramBlockWindowLengths& v_dram_block_window_lengths, // N1*K1 tile
               const VPageBlockNavigator& v_page_block_navigator,
               const VElementFunction& v_element_func,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               const BiasElementFunction& bias_element_func,
               LSEaccDramBlockWindowTmp& lse_acc_dram_window_tmp, // M0*1 tile
               const LSEaccElementFunction& lse_acc_element_func,
               const SAccElementFunction& s_acc_element_func,
               const PComputeElementFunction& p_compute_element_func,
               const OAccElementFunction& o_acc_element_func,
               index_t num_splits,
               index_t i_split,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               index_t kv_l2p_offset, // logical-to-physical offset of seqlen_k coordinate
               void* smem_ptr) const
    {
        static_assert(
            std::is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KPageBlockNavigator::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VPageBlockNavigator::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == KDramBlockWindowLengths{}[number<0>{}] &&
                          kK0 == KDramBlockWindowLengths{}[number<1>{}] &&
                          kN1 == VDramBlockWindowLengths{}[number<0>{}] &&
                          kK1 == VDramBlockWindowLengths{}[number<1>{}] &&
                          kM0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<0>{}] &&
                          kN0 == BiasDramBlockWindowTmp{}.get_window_lengths()[number<1>{}],
                      "wrong!");

        // K tile in LDS
        KDataType* k_lds_ptr = static_cast<KDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeQ<Problem>()));
        auto k_lds           = make_tensor_view<address_space_enum::lds>(
            k_lds_ptr, Policy::template MakeKLdsBlockDescriptor<Problem>());
        auto k_lds_window =
            make_tile_window(k_lds, make_tuple(number<kN0>{}, number<kK0>{}), {0, 0});

        // V tile in LDS
        auto v_lds = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<VDataType*>(smem_ptr),
            Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_window = make_tile_window(
            v_lds, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetKVBlockGemm<Problem>();

        auto q_dram_window = make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
                                              q_dram_block_window_tmp.get_window_lengths(),
                                              q_dram_block_window_tmp.get_window_origin(),
                                              Policy::template MakeQRegTileDistribution<Problem>());

        auto q = load_tile(q_dram_window);

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

        const auto q_origin          = q_dram_window.get_window_origin();
        const auto tile_range_result = [&mask, &q_origin, num_splits, i_split]() {
            if constexpr(kHasSink)
                return mask.GetSinkTileRangeAlongX(
                    q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{}, num_splits, i_split);
            else
            {
                auto [start, end] = mask.GetTileRangeAlongX(
                    q_origin.at(number<0>{}), number<kM0>{}, number<kN0>{}, num_splits, i_split);
                return ck_tile::make_tuple(0, start, end);
            }
        }();
        const auto sink_seq_end           = tile_range_result.get(ck_tile::number<0>{});
        const auto logical_seqlen_k_start = tile_range_result.get(ck_tile::number<1>{});
        const auto logical_seqlen_k_end   = tile_range_result.get(ck_tile::number<2>{});

        const auto num_sink_loop = integer_divide_ceil(sink_seq_end, kN0);

        // check early exit if no work to do
        if constexpr(FmhaMask::IsMasking || kPadSeqLenK || kHasUnevenSplits)
        {
            const index_t logical_num_total_loop =
                integer_divide_ceil(logical_seqlen_k_end - logical_seqlen_k_start, kN0);
            if(logical_num_total_loop <= 0)
            {
                if constexpr(kStoreLSE)
                {
                    auto lse_acc =
                        make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

                    set_tile(lse_acc, -numeric<SMPLComputeDataType>::infinity());

                    store_tile(lse_acc_dram_window_tmp,
                               tile_elementwise_in(lse_acc_element_func, lse_acc));
                }

                // Note: here occ are all cleard, return it
                // Note: q loaded but no fence, ignore it.
                return o_acc;
            }
        }

        const index_t physical_seqlen_k_start = logical_seqlen_k_start + kv_l2p_offset;
        const index_t physical_seqlen_k_end   = logical_seqlen_k_end + kv_l2p_offset;
        // make sure the first tile is completely located in page-block (page-block size should be
        // divisible by kN0)
        // relationship between each *_start variables: aligned_physical_seqlen_k_start <=
        // physical_seqlen_k_start, logical_seqlen_k_start <= physical_seqlen_k_start
        const index_t aligned_physical_seqlen_k_start =
            [&, physical_seqlen_k_start_ = physical_seqlen_k_start] {
                if constexpr(kIsPagedKV)
                {
                    return kN0 * integer_divide_floor(physical_seqlen_k_start_, kN0);
                }
                else
                {
                    return physical_seqlen_k_start_;
                }
            }();
        const auto kv_load_start = (sink_seq_end == 0 && aligned_physical_seqlen_k_start > 0)
                                       ? aligned_physical_seqlen_k_start
                                       : 0;
        const index_t num_total_loop =
            integer_divide_ceil(physical_seqlen_k_end - aligned_physical_seqlen_k_start, kN0) +
            num_sink_loop;

        auto [i_page_block_k, k_dram_block_window] = k_page_block_navigator.make_tile_window(
            k_dram_block_window_lengths, {kv_load_start, 0});

        const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();

        const index_t bias_n_offset = [&]() {
            if constexpr(kHasSink)
                return kv_load_start;
            else
                return logical_seqlen_k_start -
                       (physical_seqlen_k_start - aligned_physical_seqlen_k_start);
        }();

        auto bias_dram_window =
            make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                             bias_dram_block_window_tmp.get_window_lengths(),
                             {bias_origin.at(number<0>{}), bias_n_offset},
                             Policy::template MakeBiasDramTileDistribution<decltype(gemm_0)>());

        auto [i_page_block_v, v_dram_window] = v_page_block_navigator.make_tile_window(
            v_dram_block_window_lengths,
            {0, kv_load_start}, // TODO: hdim split?
            Policy::template MakeVDramTileDistribution<Problem>());

        auto q_tile = tile_elementwise_in(q_element_func, q);

        // prefetch K tile
        index_t i_total_loops      = 0;
        constexpr index_t k0_loops = kQKHeaddim / kK0;
        constexpr index_t k1_loops = kN0 / kK1;

        static_assert(2 <= k0_loops);
        static_assert(1 <= k1_loops);
        do
        {
            // STAGE 1, QK gemm
            auto k_dram_window = make_tile_window(
                k_dram_block_window,
                Policy::template MakeKDramTileDistribution<Problem>()); // K DRAM tile window for
                                                                        // load

            auto k_block_tile = load_tile(k_dram_window);
            {
                // moving k_dram_window is an in-page-block operation, so there is
                // no need to invoke k_page_block_navigator.move_tile_window() here.
                move_tile_window(k_dram_window, {0, kK0});
                clear_tile(s_acc); // initialize C
                store_tile(k_lds_window, tile_elementwise_in(k_element_func, k_block_tile));
                k_block_tile = load_tile(k_dram_window);
            }
            const bool is_sink_tile = ((num_sink_loop - 1) == i_total_loops);

            const auto k_move_offset = [&]() {
                if constexpr(kHasSink)
                    return is_sink_tile ? logical_seqlen_k_start - sink_seq_end + kN0 : kN0;
                else
                    return kN0;
            }();

            auto physical_next_block_id_k =
                amd_wave_read_first_lane(k_page_block_navigator.prefetch_table_id(
                    i_page_block_k, k_dram_block_window, {k_move_offset, 0}));
            auto physical_next_block_id_v = amd_wave_read_first_lane(
                v_page_block_navigator.prefetch_table_id(i_page_block_v, v_dram_window, {0, kK1}));

            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                __builtin_amdgcn_sched_barrier(
                    0); // prevent from messing up the order of global loads
            }
            const auto bias_tile = load_tile(bias_dram_window); // load bias tile
            if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS)
            {
                __builtin_amdgcn_sched_barrier(
                    0); // prevent from messing up the order of global loads
            }

            if constexpr(k0_loops > 2)
            {
                static_for<0, k0_loops - 2, 1>{}([&](auto i_k0) {
                    block_sync_lds();
                    gemm_0(s_acc,
                           get_slice_tile(q_tile,
                                          sequence<0, i_k0 * kK0>{},
                                          sequence<kM0, (i_k0 + 1) * kK0>{}),
                           k_lds_window);
                    block_sync_lds();
                    move_tile_window(k_dram_window, {0, kK0});

                    store_tile(
                        k_lds_window,
                        tile_elementwise_in(k_element_func, k_block_tile)); // LDS write i + 1
                    k_block_tile = load_tile(k_dram_window);                // global read i + 2
                });
            }

            const auto v_prefetch = load_tile(v_dram_window); // prefetch load v tile
            {                                                 // tail
                block_sync_lds();
                gemm_0(s_acc,
                       get_slice_tile(q_tile,
                                      sequence<0, (k0_loops - 2) * kK0>{},
                                      sequence<kM0, (k0_loops - 1) * kK0>{}),
                       k_lds_window);
                block_sync_lds();

                store_tile(k_lds_window, tile_elementwise_in(k_element_func, k_block_tile));
                block_sync_lds();

                gemm_0(s_acc,
                       get_slice_tile(q_tile,
                                      sequence<0, (k0_loops - 1) * kK0>{},
                                      sequence<kM0, k0_loops * kK0>{}),
                       k_lds_window);
            }

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
                const auto k_origin = k_page_block_navigator.to_global_window_origin(
                    i_page_block_k, k_dram_block_window.get_window_origin());
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
                        // position_encoding accept only logical coordinates, do conversion here
                        position_encoding.update(s_acc(i_j_idx), row, col - kv_l2p_offset);
                    });
                });
            }
            else
            {
                s_acc = tile_elementwise_in(s_acc_element_func, s_acc);
                if constexpr(kHasLogitsSoftCap)
                {
                    auto apply_logits_transform =
                        [&variant, &variant_params, &block_indices](auto& x) {
                            x = variant.LogitsTransform(variant_params,
                                                        variant.QueryTransform(variant_params, x),
                                                        block_indices.batch_idx,
                                                        block_indices.qo_head_idx,
                                                        block_indices.kv_head_idx);
                        };
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                    tile_elementwise_inout(apply_logits_transform, s_acc);
#else
                    tile_elementwise_inout(apply_logits_transform, s_acc);
#endif
                }
                else
                {
#if !CK_TILE_FMHA_FWD_FAST_EXP2
                    tile_elementwise_inout([&scale_s](auto& x) { x = x * scale_s; }, s_acc);
#endif
                }
            }
            move_tile_window(bias_dram_window, {0, k_move_offset});

            /// TODO: only check in first/last iteration without increasing code size
            if constexpr(kHasUnevenSplits)
            {
                const auto k_origin = k_page_block_navigator.to_global_window_origin(
                    i_page_block_k, k_dram_block_window.get_window_origin());
                set_tile_if(
                    s_acc,
                    -numeric<SMPLComputeDataType>::infinity(),
                    [&,
                     physical_seqlen_k_start_ = is_sink_tile ? 0 : physical_seqlen_k_start,
                     physical_seqlen_k_end_   = physical_seqlen_k_end](auto tile_idx) {
                        const auto col = k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                        if constexpr(kIsPagedKV)
                        {
                            return col < physical_seqlen_k_start_ || physical_seqlen_k_end_ <= col;
                        }
                        else
                        {
                            return physical_seqlen_k_end_ <= col;
                        }
                    });
            }

            if constexpr(kPadSeqLenK || FmhaMask::IsMasking)
            {
                const auto k_origin = k_page_block_navigator.to_global_window_origin(
                    i_page_block_k, k_dram_block_window.get_window_origin());
                // mask accept only logical coordinates, do conversion here
                bool need_perpixel_check = mask.IsEdgeTile(q_origin.at(number<0>{}),
                                                           k_origin.at(number<0>{}) - kv_l2p_offset,
                                                           number<kM0>{},
                                                           number<kN0>{});
                if(need_perpixel_check)
                {
                    auto apply_mask = [&](auto&& mask_func) {
                        set_tile_if(
                            s_acc, -numeric<SMPLComputeDataType>::infinity(), [&](auto tile_idx) {
                                const auto row =
                                    q_origin.at(number<0>{}) + tile_idx.at(number<0>{});
                                const auto col =
                                    k_origin.at(number<0>{}) + tile_idx.at(number<1>{});
                                return mask_func(row, col - kv_l2p_offset);
                            });
                    };

                    if constexpr(kHasSink)
                    {
                        apply_mask(
                            [&](auto row, auto col) { return mask.IsOutOfSinkBound(row, col); });
                    }
                    else
                    {
                        apply_mask([&](auto row, auto col) { return mask.IsOutOfBound(row, col); });
                    }
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
                        if constexpr(kHasLogitsSoftCap)
                        {
                            p_compute(i_j_idx) = exp2(s[i_j_idx] - get_validated_m(m[i_idx]));
                        }
                        else
                        {
                            p_compute(i_j_idx) = exp2(scale_s * s[i_j_idx] - row_max);
                        }
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
                        if constexpr(kHasLogitsSoftCap)
                        {

                            return exp2(m_old[i_idx] - get_validated_m(m[i_idx]));
                        }
                        else
                        {
                            auto row_max = scale_s * get_validated_m(m[i_idx]);
                            return exp2(scale_s * m_old[i_idx] - row_max);
                        }
                    }
                }();
#else
                const auto tmp       = exp(m_old[i_idx] - get_validated_m(m[i_idx]));
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

            block_sync_lds();
            if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
            {
                auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                    Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                shuffle_tile(v_shuffle_tmp, v_prefetch);
                store_tile(
                    v_lds_window,
                    tile_elementwise_in(v_element_func, v_shuffle_tmp)); // store the prefetch
            }
            else
            {
                store_tile(v_lds_window,
                           tile_elementwise_in(v_element_func, v_prefetch)); // store the prefetch
            }
            i_page_block_v = v_page_block_navigator.move_tile_window(
                i_page_block_v, v_dram_window, {0, kK1}, physical_next_block_id_v);

            const auto p =
                cast_tile<PDataType>(tile_elementwise_in(p_compute_element_func, p_compute));

            // STAGE 3, KV gemm
            if constexpr(k1_loops > 1)
            {
                static_for<0, k1_loops - 1, 1>{}([&,
                                                  &i_page_block_v_ = i_page_block_v,
                                                  &v_dram_window_  = v_dram_window](auto i_k1) {
                    auto physical_next_block_id_v_ =
                        amd_wave_read_first_lane(v_page_block_navigator.prefetch_table_id(
                            i_page_block_v_, v_dram_window_, {0, kK1}));
                    const auto v = load_tile(v_dram_window_); // load next v
                    block_sync_lds();
                    gemm_1(o_acc,
                           get_slice_tile(
                               p, sequence<0, i_k1 * kK1>{}, sequence<kM0, (i_k1 + 1) * kK1>{}),
                           v_lds_window);
                    block_sync_lds();
                    if constexpr(std::is_same_v<VLayout, ck_tile::tensor_layout::gemm::RowMajor>)
                    {
                        auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                            Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                        shuffle_tile(v_shuffle_tmp, v);
                        store_tile(v_lds_window,
                                   tile_elementwise_in(v_element_func,
                                                       v_shuffle_tmp)); // store the prefetch
                    }
                    else
                    {
                        store_tile(v_lds_window,
                                   tile_elementwise_in(v_element_func, v)); // store next v
                    }
                    i_page_block_v_ = v_page_block_navigator.move_tile_window(
                        i_page_block_v_, v_dram_window_, {0, kK1}, physical_next_block_id_v_);
                });
            }
            // move K tile windows
            i_page_block_k = k_page_block_navigator.move_tile_window(
                i_page_block_k, k_dram_block_window, {k_move_offset, 0}, physical_next_block_id_k);
            physical_next_block_id_v =
                amd_wave_read_first_lane(v_page_block_navigator.prefetch_table_id(
                    i_page_block_v, v_dram_window, {0, k_move_offset - kN0}));
            i_page_block_v = v_page_block_navigator.move_tile_window(
                i_page_block_v, v_dram_window, {0, k_move_offset - kN0}, physical_next_block_id_v);
            // tail
            {
                block_sync_lds();
                gemm_1(o_acc,
                       get_slice_tile(p, sequence<0, (k1_loops - 1) * kK1>{}, sequence<kM0, kN0>{}),
                       v_lds_window);
                block_sync_lds();
            }
        } while(++i_total_loops < num_total_loop);

        if constexpr(kStoreLSE)
        {
            // store lse acc
            auto lse_acc = make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());

            constexpr auto lse_acc_spans = decltype(lse_acc)::get_distributed_spans();
            sweep_tile_span(lse_acc_spans[number<0>{}], [&, m_ = m, l_ = l](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
#if CK_TILE_FMHA_FWD_FAST_EXP2
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             BiasEnum == BlockAttentionBiasEnum::ALIBI)
                {
                    lse_acc(i_idx) = m_[i_idx] / C_LOG2E + log(l_[i_idx]);
                }
                else
                {
                    if constexpr(kHasLogitsSoftCap)
                    {
                        lse_acc(i_idx) = m_[i_idx] / C_LOG2E + log(l_[i_idx]);
                    }
                    else
                    {
                        lse_acc(i_idx) = m_[i_idx] * scale_s / C_LOG2E + log(l_[i_idx]);
                    }
                }
#else
                    lse_acc(i_idx) = m_[i_idx] + log(l_[i_idx]);
#endif
            });

            store_tile(lse_acc_dram_window_tmp, tile_elementwise_in(lse_acc_element_func, lse_acc));
        }

        // finally, O
        constexpr auto o_spans = decltype(o_acc)::get_distributed_spans();

        sweep_tile_span(o_spans[number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const auto tmp       = [&]() {
                if constexpr(BiasEnum == BlockAttentionBiasEnum::ELEMENTWISE_BIAS ||
                             FmhaMask::IsMasking)
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
              typename KDramBlockWindowLengths,
              typename KPageBlockNavigator,
              typename VDramBlockWindowLengths,
              typename VPageBlockNavigator,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename PositionEncoding,
              typename AttentionVariantParams,
              typename BlockIndices>
    CK_TILE_HOST_DEVICE auto
    operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,         // M0*K0 tile
               const KDramBlockWindowLengths& k_dram_block_window_lengths, // N0*K0 tile
               const KPageBlockNavigator& k_page_block_navigator,
               const VDramBlockWindowLengths& v_dram_block_window_lengths, // N1*K1 tile
               const VPageBlockNavigator& v_page_block_navigator,
               const BiasDramBlockWindowTmp& bias_dram_block_window_tmp, // M0*N0 tile
               LSEaccDramBlockWindowTmp& lse_acc_dram_block_window_tmp,  // M0*1 tile
               index_t num_splits,
               index_t i_split,
               FmhaMask mask,
               PositionEncoding position_encoding,
               float scale_s,
               const AttentionVariant& variant,
               const AttentionVariantParams& variant_params,
               const BlockIndices& block_indices,
               index_t kv_l2p_offset, // logical-to-physical offset of seqlen_k coordinate
               void* smem_ptr) const
    {
        return operator()(q_dram_block_window_tmp,
                          identity{},
                          k_dram_block_window_lengths,
                          k_page_block_navigator,
                          identity{},
                          v_dram_block_window_lengths,
                          v_page_block_navigator,
                          identity{},
                          bias_dram_block_window_tmp,
                          identity{},
                          lse_acc_dram_block_window_tmp,
                          identity{},
                          identity{},
                          identity{},
                          identity{},
                          num_splits,
                          i_split,
                          mask,
                          position_encoding,
                          scale_s,
                          variant,
                          variant_params,
                          block_indices,
                          kv_l2p_offset,
                          smem_ptr);
    }
};

} // namespace ck_tile
