// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/fused_moe/pipeline/fused_moegemm_pipeline_flatmm_policy.hpp"

namespace ck_tile {

/*
This pipeline deal with a gemm(actually 2 gemm) with one very small(token), one very big(weight)
we need to design the pipeline such that all waves along gemm-N dim (gemm-m only 1 wave)

    <----- gemm-N ------>
    +----+----+----+----+
    | w0 | w1 | w2 | w3 | gemm-m
    +----+----+----+----+
*/
template <typename Problem_, typename Policy_ = FusedMoeGemmPipelineFlatmmPolicy>
struct FusedMoeGemmPipeline_FlatmmUk
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using BlockShape = typename Problem::BlockShape; // this is FusedMoeGemmShape

    using ADataType            = typename Problem::ADataType;
    using GDataType            = typename Problem::GDataType;
    using DDataType            = typename Problem::DDataType;
    using AccDataType          = typename Problem::AccDataType;
    using ODataType            = typename Problem::ODataType;
    using AScaleDataType       = typename Problem::AScaleDataType;
    using GScaleDataType       = typename Problem::GScaleDataType;
    using DScaleDataType       = typename Problem::DScaleDataType;
    using YSmoothScaleDataType = typename Problem::YSmoothScaleDataType;
    using TopkWeightDataType   = typename Problem::TopkWeightDataType;
    using IndexDataType        = typename Problem::IndexDataType;
    using YDataType            = typename Problem::YDataType;

    using Traits = typename Problem::Traits;

    static constexpr bool IsGateOnly          = Traits::IsGateOnly;
    static constexpr bool UseSmoothQuant      = Traits::UseSmoothQuant;
    static constexpr bool PadHiddenSize       = Traits::PadHiddenSize;
    static constexpr bool PadIntermediateSize = Traits::PadIntermediateSize;

    static constexpr index_t kAlignmentA = Policy::template GetAlignment_A<Problem>();
    static constexpr index_t kAlignmentG = Policy::template GetAlignment_G<Problem>();
    static constexpr index_t kAlignmentD = Policy::template GetAlignment_D<Problem>();
    static constexpr index_t kAlignmentO = Policy::template GetAlignment_O<Problem>();

    static constexpr index_t SLD_A = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::SLD_A);
    static constexpr index_t GLD_A = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GLD_A);
    static constexpr index_t GLD_B = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GLD_B);
    static constexpr index_t GST_O = static_cast<index_t>(FusedMoeGemmPipelineSequencerEnum::GST_O);

    static constexpr index_t kBlockPerCu = []() {
        if constexpr(Problem::kBlockPerCu != -1)
            return Problem::kBlockPerCu;
        else
        {
            // minimize occupancy
            return 2;
        }
    }();

    static constexpr const char* name = "flatmm_uk";

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
#if 1
        constexpr index_t smem_0 = Policy::template GetUK_0<Problem>().GetSmemSize();
        constexpr index_t smem_1 = Policy::template GetUK_1<Problem>().GetSmemSize();
        constexpr index_t smem_bridge =
            BlockShape::Block_M0 * BlockShape::Block_N0 * sizeof(YDataType);
        return max(smem_0 + smem_1, smem_bridge);
#else
        // keep it here purposely in case we have regression
        return 65536;
#endif
    }

    // this is the thread-offset along row/col
    CK_TILE_HOST_DEVICE static auto GetACoord()
    {
        constexpr auto a_dist = Policy::template MakeGlobalTileDistribution_A<Problem>();
        const auto a_coord    = a_dist.calculate_index();
        return a_coord;
    }

    // this is the thread-offset along row/col
    CK_TILE_HOST_DEVICE static auto GetOCoord()
    {
        constexpr auto o_dist = Policy::template MakeOGlobalTileDistribution<Problem>();
        const auto o_coord    = o_dist.calculate_index();
        return o_coord;
    }

    CK_TILE_DEVICE constexpr auto GetNumRowCoords_A()
    {
        constexpr index_t KLans   = BlockShape::Block_K0 / kAlignmentA;
        constexpr index_t MLans   = BlockShape::BlockSize / KLans;
        constexpr index_t MRepeat = BlockShape::Block_M0 / MLans;

        return MRepeat;
    }

    // TODO: properlly support scatter/gather
    CK_TILE_DEVICE auto GetRowCoords_A(index_t base_offset)
    {
        constexpr index_t KLans   = BlockShape::Block_K0 / kAlignmentA;
        constexpr index_t MLans   = BlockShape::BlockSize / KLans;
        constexpr index_t MRepeat = BlockShape::Block_M0 / MLans;

        auto base_coord = threadIdx.x / KLans + base_offset;

        array<index_t, MRepeat> coords;
        static_for<0, MRepeat, 1>{}([&](auto i) { coords.at(i) = base_coord + i * MLans; });

        return coords;
    }

    template <typename ROW_COORDS>
    CK_TILE_DEVICE auto GetRowID(const ROW_COORDS coords, const IndexDataType* sorted_token_ids_ptr)
    {
        constexpr index_t n_size = coords.size();

        array<index_t, n_size> row_ids;
        static_for<0, n_size, 1>{}([&](auto i) {
            row_ids.at(i) = sorted_token_ids_ptr[coords[i]]; // base_coord + i * MLans;
#if CK_TILE_REFERENCE_MOE_SORTING_MOCK_ID
            row_ids.at(i) &= 0xffffff;
#endif
        });

        return row_ids;
    }

    template <typename ROW_COORDS>
    CK_TILE_DEVICE auto GetWeightScale(const ROW_COORDS coords,
                                       const TopkWeightDataType* sorted_weight_ptr)
    {
        constexpr index_t n_size = coords.size();

        array<TopkWeightDataType, n_size> w;
        static_for<0, n_size, 1>{}([&](auto i) {
            w.at(i) = sorted_weight_ptr[coords[i]]; // base_coord + i * MLans;
        });

        return w;
    }

    // TODO: this row id is before shuffle atomic, need use acc distribution
    CK_TILE_DEVICE auto GetRowCoords_O(index_t base_offset)
    {
        constexpr index_t MLanes   = BlockShape::Warp_M1;
        constexpr index_t Repeat_M = BlockShape::Repeat_M1;

        auto base_coord = threadIdx.x % MLanes + base_offset;

        array<index_t, Repeat_M> coords;
        static_for<0, Repeat_M, 1>{}([&](auto i) { coords.at(i) = base_coord + i * MLanes; });

        return coords;
    }

    template <typename Karg>
    CK_TILE_DEVICE auto operator()(const Karg& kargs,
                                   CK_TILE_LDS_ADDR void* smem,
                                   index_t sorted_tile_id,
                                   index_t intermediate_tile_id)
    {
        constexpr index_t hidden_radio_0 = IsGateOnly ? 1 : 2;
        ck_tile::index_t shared_intermediate_size_0 =
            kargs.intermediate_size * hidden_radio_0; // total gate+up
        ck_tile::index_t shared_intermediate_size_1 = kargs.intermediate_size;

        // after weight shuffling, gate-only: [nr0, kr0, w0], gate+up: [nr0_gate + nr0_up, kr0, w0]

        index_t nr_0 = shared_intermediate_size_0 / BlockShape::Warp_N0; // divide N in W
        index_t kr_0 = kargs.hidden_size / BlockShape::Warp_K0;          // divide K in W
        index_t nr_1 = kargs.hidden_size / BlockShape::Warp_N1;
        index_t kr_1 = shared_intermediate_size_1 / BlockShape::Warp_K1;

        const IndexDataType expert_id = amd_wave_read_first_lane(
            reinterpret_cast<const IndexDataType*>(kargs.sorted_expert_ids_ptr)[sorted_tile_id]);
        index_t expert_stride_0 = shared_intermediate_size_0 * kargs.hidden_size;
        index_t expert_stride_1 = shared_intermediate_size_1 * kargs.hidden_size;

        // nr*kr*w
        index_t interm_idx_nr0 = amd_wave_read_first_lane(
            intermediate_tile_id *
            BlockShape::Block_Nr0); // intermediate_tile_id * Block_N / (N in W)

        index_t interm_idx_kr1 = amd_wave_read_first_lane(
            intermediate_tile_id *
            BlockShape::Block_Kr1); // intermediate_tile_id * Block_N / (N in W)

        auto row_coords_a = GetRowCoords_A(sorted_tile_id * BlockShape::Block_M0);
        auto row_ids_a    = GetRowID(
            row_coords_a, reinterpret_cast<const IndexDataType*>(kargs.sorted_token_ids_ptr));
        auto a_coords = generate_tuple(
            [&](auto i) {
                return row_ids_a[i] * kargs.stride_token +
                       threadIdx.x % (BlockShape::Block_K0 / kAlignmentA) * kAlignmentA;
            },
            number<row_ids_a.size()>{});

        auto a_res =
            make_wave_buffer_resource(reinterpret_cast<const ADataType*>(kargs.a_ptr),
                                      kargs.num_tokens * kargs.stride_token * sizeof(ADataType),
                                      std::true_type{});

        auto make_gu_win = [&](const auto* ptr_) {
            auto view_ = make_naive_tensor_view<address_space_enum::global>(
                ptr_,
                make_tuple(nr_0, kr_0, number<BlockShape::Block_W0>{}),
                make_tuple(kr_0 * BlockShape::Block_W0, number<BlockShape::Block_W0>{}, 1),
                number<kAlignmentG>{},
                number<1>{});

            auto win_ = make_tile_window_linear_raw(
                view_,
                make_tuple(number<BlockShape::Block_Nr0>{},
                           number<BlockShape::Block_Kr0>{},
                           number<BlockShape::Block_W0>{}),
                {0, 0, 0},
                Policy::template MakeGlobalTileDistribution_G<Problem>(),
                sequence<0, 1, 1>{});
            return win_;
        };

        const GDataType* gu_ptr = reinterpret_cast<const GDataType*>(kargs.g_ptr) +
                                  static_cast<long_index_t>(expert_id) * expert_stride_0 +
                                  interm_idx_nr0 * kr_0 * BlockShape::Block_W0;

        auto g_win = make_gu_win(gu_ptr);
        // Note: gu swizzled, [nr_u+nr_g, kr, w], hence base offset to up is just interm*hidden
        auto u_win = make_gu_win(gu_ptr + kargs.intermediate_size * kargs.hidden_size);

        auto g_res    = g_win.get_bottom_tensor_view().get_buffer_view().cached_buf_res_;
        auto u_res    = u_win.get_bottom_tensor_view().get_buffer_view().cached_buf_res_;
        auto g_coords = generate_tuple([&](auto i) { return g_win.cached_coords_[i].get_offset(); },
                                       number<decltype(g_win)::NumAccess_NonLinear>{});

        const auto d_win = [&]() {
            const DDataType* d_ptr = reinterpret_cast<const DDataType*>(kargs.d_ptr) +
                                     static_cast<long_index_t>(expert_id) * expert_stride_1 +
                                     interm_idx_kr1 * BlockShape::Block_W1;
            // note interm_idx_nr0 is along the gemm-k dim of 2nd gemm

            const auto d_view_ = make_naive_tensor_view<address_space_enum::global>(
                d_ptr,
                make_tuple(nr_1, kr_1, BlockShape::Block_W1),
                make_tuple(kr_1 * BlockShape::Block_W1, BlockShape::Block_W1, 1),
                number<kAlignmentD>{},
                number<1>{});

            const auto d_window_ = make_tile_window_linear_raw(
                d_view_,
                make_tuple(number<BlockShape::Block_Nr1>{},
                           number<BlockShape::Block_Kr1>{},
                           number<BlockShape::Block_W1>{}),
                {0, 0, 0},
                Policy::template MakeGlobalTileDistribution_D<Problem>(),
                sequence<0, 1, 1>{});
            return d_window_;
        }();
        auto d_res = d_win.get_bottom_tensor_view().get_buffer_view().cached_buf_res_;

        // TODO: load D order is N0.K0...127, N64.K0...127, N0.K128...255, N64.K128...255
        //      block-k=512, block-n=128
        //                    wg                     |<----- W_   ----->|
        //       Nr(2)*Nw(4)* Kr *Kr0(4)*Kr1(4) * [Kl(4)*Nl(16)*Kv(8)]->one issue
        //          y   p          y     y         p     p       y
        //          1              2     0(imm)
        auto d_coords = [&]() {
            constexpr index_t Nr_          = 2;
            constexpr index_t Nw_          = 4;
            constexpr index_t Kr0_         = 4;
            constexpr index_t Kr1_         = 4;
            constexpr index_t Kl_          = 4;
            constexpr index_t Nl_          = 16;
            constexpr index_t Kv_          = 8;
            constexpr index_t W_           = Kl_ * Nl_ * Kv_;
            constexpr index_t num_offsets_ = Nr_ * Kr0_;
            index_t base_os_               = (threadIdx.x % 64) * Kv_ + (threadIdx.x / 64) *
                                                              shared_intermediate_size_1 *
                                                              Nl_; // Kr0_ * Kr1_ * W_;
            return generate_tuple(
                [&](auto i) {
                    constexpr auto i_nr_  = number<i % Nr_>{};
                    constexpr auto i_kr0_ = number<i / Nr_>{};

                    return i_nr_ * shared_intermediate_size_1 * Nw_ * Nl_ + i_kr0_ * Kr1_ * W_ +
                           base_os_;
                },
                number<num_offsets_>{});
        }();

        auto o_coords = generate_tuple(
            [&](auto i) {
                return row_ids_a[i] * kargs.stride_token +
                       threadIdx.x % (BlockShape::Block_N1 / kAlignmentO) * kAlignmentO;
            },
            number<row_ids_a.size()>{});

        auto o_flags =
            generate_tuple([&](auto i) { return cmp_lt_to_exec(row_ids_a[i], kargs.num_tokens); },
                           number<row_ids_a.size()>{});

        auto bridge_sst_win = [&]() {
            constexpr auto desc_ = Policy::template MakeBridgeLdsStoreForUKDesc<Problem>();
            constexpr auto dist_ = Policy::template GetUK_0<Problem>().MakeCBlockDist();
            return make_tile_window_linear(make_tensor_view<address_space_enum::lds>(
                                               reinterpret_cast<YDataType*>(smem), desc_),
                                           desc_.get_lengths(),
                                           {0, 0},
                                           dist_);
        }();

        auto o_res =
            make_wave_buffer_resource(reinterpret_cast<const ODataType*>(kargs.o_ptr),
                                      kargs.num_tokens * kargs.stride_token * sizeof(ODataType),
                                      std::true_type{});
        auto row_coords_o = GetRowCoords_O(sorted_tile_id * BlockShape::Block_M0);
        auto w_scale      = GetWeightScale(
            row_coords_o, reinterpret_cast<const TopkWeightDataType*>(kargs.sorted_weight_ptr));

        auto uk_0 = Policy::template GetUK_0<Problem>();

        auto y_pre = [&]() {
            if constexpr(IsGateOnly)
            {
                auto acc_0 = uk_0(a_res,
                                  a_coords,
                                  g_res,
                                  g_coords,
                                  smem,
                                  kargs.hidden_size,
                                  BlockShape::Block_K0, // tile offset for B matrix each unroll
                                  BlockShape::Block_Kr0 *
                                      BlockShape::Block_W0); // tile offset for B matrix each unroll

                sweep_tile(
                    acc_0,
                    [&](auto idx0, auto idx1) {
                        fp32x2_t v_{acc_0(idx0), acc_0(idx1)};
                        typename Problem::GateActivation{}(v_, v_);
                        acc_0(idx0) = v_.x;
                        acc_0(idx1) = v_.y;
                    },
                    sequence<1, 2>{});

                return cast_tile<YDataType>(acc_0);
            }
            else
            {
                uint32x8_t gu_res;
                gu_res[0] = g_res[0];
                gu_res[1] = g_res[1];
                gu_res[2] = g_res[2];
                gu_res[3] = g_res[3];
                gu_res[4] = u_res[0];
                gu_res[5] = u_res[1];
                gu_res[6] = u_res[2];
                gu_res[7] = u_res[3];

                auto acc_0 = uk_0(a_res,
                                  a_coords,
                                  gu_res,
                                  g_coords,
                                  smem,
                                  kargs.hidden_size,
                                  BlockShape::Block_K0, // tile offset for B matrix each unroll
                                  BlockShape::Block_Kr0 * BlockShape::Block_W0,
                                  bool_constant<true>{}); // tile offset for B matrix each unroll

                sweep_tile(
                    acc_0.at(number<0>{}),
                    [&](auto idx0, auto idx1) {
                        fp32x2_t v_{acc_0.at(number<0>{})(idx0), acc_0.at(number<0>{})(idx1)};
                        typename Problem::GateActivation{}(v_, v_);
                        acc_0.at(number<0>{})(idx0) = v_.x;
                        acc_0.at(number<0>{})(idx1) = v_.y;
                    },
                    sequence<1, 2>{});

                auto reduced_acc_0 =
                    tile_elementwise_in([&](const auto& a_, const auto& b_) { return a_ * b_; },
                                        acc_0.at(number<0>{}),
                                        acc_0.at(number<1>{}));

                return cast_tile<YDataType>(reduced_acc_0);
            }
        }();

        block_sync_lds();

        store_tile(bridge_sst_win, y_pre);
        block_sync_lds();

        auto uk_1 = Policy::template GetUK_1<Problem>();
        uk_1(d_res,
             d_coords,
             o_res,
             o_coords,
             o_flags,
             smem,
             kargs.hidden_size, // total n number
             w_scale,
             BlockShape::Block_Nr1 * kr_1 * BlockShape::Block_W1, // along N
             BlockShape::Block_N1);                               // along N
    }
};

} // namespace ck_tile
